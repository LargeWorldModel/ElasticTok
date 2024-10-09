import pprint
import os

from tqdm import tqdm, trange
import numpy as np
from absl.app import run
import wandb
import absl.logging as logging
import multiprocessing as mp

import tux
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map
from flax.training.train_state import TrainState

from elastic.data import DatasetFactory
from tux import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    global_norm, get_float_dtype_by_name,
    set_random_seed, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint, define_flags_with_default,
    OptimizerFactory, StreamingCheckpointer
)
from elastic.model import ElasticTokConfig, ElasticTok
from elastic.inference import ElasticInference
from elastic import lpips


FLAGS, FLAGS_DEF = define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    total_steps=10000,
    load_elastic_config='',
    update_elastic_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    eval_freq=50,
    eval_thresholds='0.015,0.003',
    save_model_freq=0,
    save_milestone_freq=0,
    train_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    elastic_tok=ElasticTokConfig.get_default_config(),
    logger=tux.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    autoresume=False,
)


def main(argv):
    assert FLAGS.eval_freq % FLAGS.log_freq == 0, "eval_freq must be divisible by log_freq"
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = tux.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = tux.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    print(f'Started {jax.process_index()} / {jax.process_count()}')

    logger = tux.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    if jax.process_index() == 0:
        output_dir = logger.output_dir
    else:
        output_dir = os.path.join(logger.output_dir, logger.experiment_id)

    if FLAGS.load_elastic_config != '':
        elastic_config = ElasticTokConfig.load_config(FLAGS.load_elastic_config)
        updates = ElasticTokConfig(**FLAGS.elastic_tok)
        elastic_config.update(dict(
            remat_block=updates.remat_block,
            remat_attention=updates.remat_attention,
            remat_mlp=updates.remat_mlp,
            scan_attention=updates.scan_attention,
            scan_mlp=updates.scan_mlp,
            scan_query_chunk_size=updates.scan_query_chunk_size,
            scan_key_chunk_size=updates.scan_key_chunk_size,
            scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
            scan_layers=updates.scan_layers,
            param_scan_axis=updates.param_scan_axis,
        ))
    else:
        elastic_config = ElasticTokConfig(**FLAGS.elastic_tok)

    if FLAGS.update_elastic_config != '':
        elastic_config.update(dict(eval(FLAGS.update_elastic_config)))
    elastic_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    mesh = ElasticTokConfig.get_jax_mesh(FLAGS.mesh_dim)
    node_info = ElasticTokConfig.get_ranks_and_size(mesh)

    dataset = DatasetFactory.load_dataset(
        FLAGS.train_dataset, node_info=node_info, mesh=mesh,
        elastic_config=elastic_config
    )

    if FLAGS.autoresume and tux.check_exists(output_dir):
        logging.info('Found existing output. Resuming dataset from latest checkpoint...')
        resume_path = f"{output_dir}/dataset.pkl"
        dataset.load_state_dict(tux.load_pickle(resume_path))
    elif FLAGS.load_dataset_state != '':
        dataset.load_state_dict(tux.load_pickle(FLAGS.load_dataset_state))

    model = ElasticTok(
        elastic_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(ElasticTokConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        batch = mesh.shape['dp'] * mesh.shape['fsdp']
        params = model.init(
            vision=jnp.zeros((batch, elastic_config.max_sequence_length, np.prod(elastic_config.patch_size) * 3), dtype=jnp.int32),
            encoding_mask=jnp.ones((batch, elastic_config.max_sequence_length), dtype=bool),
            attention_mask=jnp.ones((batch, elastic_config.max_sequence_length), dtype=bool),
            segment_ids=jnp.zeros((batch, elastic_config.max_sequence_length), dtype=jnp.int32),
            position_ids=jnp.zeros((batch, elastic_config.max_sequence_length), dtype=jnp.int32),
            rngs=rng_generator(elastic_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, lpips_params, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        batch['vision'] = batch['vision'].astype(jnp.float32) / 127.5 - 1
        def loss_fn(params):
            recon, metrics = model.apply(
                params, batch['vision'], batch['encoding_mask'],
                batch['attention_mask'], batch['segment_ids'], batch['position_ids'],
                rngs=rng_generator(elastic_config.rng_keys()),
            )
            recon_loss = jnp.square(recon - batch['vision']).mean()

            if elastic_config.lpips_loss_ratio > 0:
                def compute_lpips(real, fake, lpips_params, block_idx, frame_idx):
                    def _to_video_shape(x, block_idx):
                        B = 1
                        x = x[block_idx][None]
                        T, H, W, C = (
                            elastic_config.frames_per_block,
                            dataset.config.resolution, dataset.config.resolution, 3)
                        Tp, Hp, Wp = elastic_config.patch_size
                        x = x.reshape(B, T // Tp, H // Hp, W // Wp, Tp, Hp, Wp, C)
                        x = jnp.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))
                        x = x.reshape(B, T, H, W, C)
                        x = jax.image.resize(x, (B, T, 224, 224, C), 'bilinear')
                        return x
                    real, fake = _to_video_shape(real, block_idx), _to_video_shape(fake, block_idx)
                    lpips_loss = lpips_model.apply(
                        lpips_params, fake[:, frame_idx], real[:, frame_idx])
                    lpips_loss = jnp.mean(lpips_loss)
                    return lpips_loss[None]
                spec = ('dp', 'fsdp', 'tp', 'sp')
                compute_lpips = shard_map(
                    compute_lpips,
                    mesh=mesh,
                    in_specs=(
                      PS(spec), PS(spec), PS(), PS(), PS()
                    ),
                    out_specs=PS(spec)
                )
                block_size = elastic_config.max_toks
                real = batch['vision'].reshape(-1, block_size, batch['vision'].shape[-1])
                fake = recon.reshape(-1, block_size, recon.shape[-1])
                block_idx = jax.random.randint(
                    rng_generator(), (), 0, real.shape[0] // jax.device_count())
                frame_idx = jax.random.randint(
                    rng_generator(), (), 0, elastic_config.frames_per_block)
                lpips_loss = compute_lpips(
                    real, fake, lpips_params, block_idx, frame_idx)
                lpips_loss = jnp.mean(lpips_loss)

                recon_loss = recon_loss + elastic_config.lpips_loss_ratio * lpips_loss
                metrics['lpips_loss'] = lpips_loss

            loss = recon_loss + metrics['aux_loss']
            metrics['recon_loss'] = recon_loss
            return loss, metrics
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
            **metrics
        )
        return train_state, rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        ElasticTokConfig.get_partition_rules(elastic_config.scan_layers, elastic_config.param_scan_axis), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS(), PS(('dp', 'fsdp'), 'sp')),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 2),
    )

    def save_checkpoint(train_state, milestone=False):
        try:
            step = int(jax.device_get(train_state.step))
            metadata = dict(
                step=step,
                variant=variant,
                flags=flags_config_dict,
                elastic_config=elastic_config.to_dict(),
            )
            checkpointer.save_all(
                train_state=train_state,
                gather_fns=gather_fns,
                metadata=metadata,
                dataset=dataset.get_state_dict(),
                milestone=milestone,
            )
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")

    with mesh:
        train_state, restored_params = None, None

        if FLAGS.autoresume and tux.check_exists(output_dir):
            logging.info('Found existing output. Resuming model from latest checkpoint...')
            resume_path = f"trainstate::{output_dir}/streaming_train_state"
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                resume_path, train_state_shapes, shard_fns
            )
        elif FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            # restored_params = flax.core.frozen_dict.unfreeze(restored_params)
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        param_count = jax.tree.map(lambda x: np.prod(x.shape), train_state.params)
        param_count, _ = jax.tree_util.tree_flatten(param_count)
        total_params = sum(param_count)
        print(f"Total parameters: {total_params / 1e6:.3f}M")

        lpips_params = lpips.load_params()
        lpips_params = pjit(
            lambda x: x,
            in_shardings=PS(),
            out_shardings=PS(None)
        )(lpips_params)
        lpips_model = lpips.LPIPS()

        inference = ElasticInference(
            model, elastic_config, dataset.config, mesh,
            node_info, train_state_partition.params, 'binary')

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        thresholds = list(map(float, FLAGS.eval_thresholds.split(',')))
        dataset = iter(dataset)
        for step, batch in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, lpips_params, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                metrics = jax.device_get(metrics)
                log_metrics = {"step": step}
                log_metrics.update(metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.eval_freq > 0 and step % FLAGS.eval_freq == 0:
                paths = []
                log_metrics = {"step": step}
                for threshold in thresholds:
                    path, ntoks = inference(
                        train_state.params, batch, threshold, dataset.config.resolution)
                    ntoks = np.mean(jax.device_get(ntoks))
                    if path.endswith('jpg'):
                        viz = wandb.Image(path)
                    else:
                        viz = wandb.Video(path, format='mp4')
                    log_metrics.update({f"viz/recon_{threshold}": viz, f"viz/ntoks_{threshold}": ntoks})
                    paths.append(path)
                logger.log(log_metrics)
                tqdm.write("\nVAL\n" + pprint.pformat(log_metrics) + "\n")
                for path in paths:
                    os.system(f"rm {path}")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)
        print('done')


if __name__ == "__main__":
    mp.set_start_method('spawn')
    run(main)
