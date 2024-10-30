import os
import os.path as osp
from collections import namedtuple
import math
import types
import numpy as np
import pickle
from absl.app import run

import jax
import jax.numpy as jnp

from tux import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    get_float_dtype_by_name,
    set_random_seed,
    make_shard_and_gather_fns, define_flags_with_default,
    StreamingCheckpointer
)
from elastic.data import _fetch, IMAGE_EXT
from elastic.model import ElasticTokConfig, ElasticTok
from elastic.inference import ElasticInference, extract_and_reshape
from elastic.vision_utils import save


FLAGS, FLAGS_DEF = define_flags_with_default(
    input_path='',
    output_folder='',
    resolution=256,
    max_blocks_per_chunk=4,
    fps=24,
    n_codes=512,
    seed=42,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    load_elastic_config='',
    update_elastic_config='',
    load_checkpoint='',
    checkpointer=StreamingCheckpointer.get_default_config(),
    elastic_tok=ElasticTokConfig.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


State = namedtuple('State', ['params'])


def main(argv):
    assert FLAGS.input_path != ''
    assert FLAGS.output_folder != ''
    os.makedirs(FLAGS.output_folder, exist_ok=True)
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    print(f'Started {jax.process_index()} / {jax.process_count()}')
    set_random_seed(FLAGS.seed)

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
    assert FLAGS.max_blocks_per_chunk * elastic_config.max_toks <= elastic_config.max_sequence_length

    mesh = ElasticTokConfig.get_jax_mesh(FLAGS.mesh_dim)
    node_info = ElasticTokConfig.get_ranks_and_size(mesh)

    model = ElasticTok(
        elastic_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

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
        return State(params)

    param_shapes = jax.eval_shape(init_fn, next_rng())
    param_partition = match_partition_rules(
        ElasticTokConfig.get_partition_rules(elastic_config.scan_layers, elastic_config.param_scan_axis),
        param_shapes
    )

    shard_fns, _ = make_shard_and_gather_fns(
        param_partition, param_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, None,
        enable=jax.process_index() == 0,
    )

    data_config = types.SimpleNamespace(
        elastic_config=elastic_config,
        resolution=FLAGS.resolution,
        fps=FLAGS.fps)
    inference = ElasticInference(
        model, elastic_config, data_config, mesh, node_info,
        param_partition.params, FLAGS.search_alg
    )

    is_image = any([FLAGS.input_path.endswith(ext) for ext in IMAGE_EXT])
    data = _fetch(FLAGS.input_path, data_config)
    n_blocks = len(data)
    if is_image:
        assert n_blocks == 1
    n_chunks = int(math.ceil(n_blocks / FLAGS.max_blocks_per_chunk))
    print(n_blocks)

    def get_batch(data):
        n_blocks = len(data)
        assert n_blocks <= FLAGS.max_blocks_per_chunk
        block_size = elastic_config.max_toks
        batch = dict(
            vision=np.concatenate(data, axis=0),
            segment_ids=np.zeros((n_blocks * block_size,), dtype=np.int32),
            attention_mask=np.ones((n_blocks * block_size,), dtype=bool),
            position_ids=np.arange(n_blocks * block_size, dtype=np.int32)
        )
        batch = {k: v[None] for k, v in batch.items()}
        return batch

    print(f"Threshold: {FLAGS.threshold}")
    with mesh:
        params = checkpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, param_shapes, shard_fns
        )[1].unfreeze()
        print('Loaded model')

        recons, gts, zs = [], [], []
        for i in range(n_chunks):
            batch = get_batch(data[i * FLAGS.max_blocks_per_chunk:(i + 1) * FLAGS.max_blocks_per_chunk])
            z = inference.encode(params, batch, FLAGS.n_codes)
            zs.extend(z[0])
            print([z.shape for z in zs])
        pickle.dump(zs, open(osp.join(FLAGS.output_folder, 'encodings.pkl'), 'wb'))
    print('Done')


if __name__ == "__main__":
    run(main)
