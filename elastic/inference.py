from functools import partial, cached_property
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from tux import with_sharding_constraint
from elastic.data import MaskSampler
from elastic.vision_utils import save_grid


def extract_codes(z, encoding_mask, block_size):
    z, encoding_mask = jax.device_get((z, encoding_mask))
    B, L = z.shape[:2]
    n_blocks = L // block_size
    encoding_mask = encoding_mask.reshape(B, n_blocks, block_size)
    n_toks = encoding_mask.sum(-1)
    out = []
    for b in range(B):
      out_i = []
      for i in range(n_blocks):
        data = z[b, i*block_size:i*block_size+n_toks[b, i]]
        out_i.append(data)
      out.append(out_i)
    return out


def extract_and_reshape(vision, resolution, elastic_config, n=16):
    B, L = vision.shape[:2]
    n_frames = L // elastic_config.max_toks * elastic_config.frames_per_block
    T, H, W = n_frames, resolution, resolution
    Tp, Hp, Wp = elastic_config.patch_size
    vision = vision.reshape(B, T // Tp, H // Hp, W // Wp, Tp, Hp, Wp, 3)
    vision = np.transpose(vision, (0, 1, 4, 2, 5, 3, 6, 7))
    vision = vision.reshape(B, T, H, W, 3)
    vision = vision[:n]
    return vision


class ElasticInference(object):
    def __init__(self, model, elastic_config, data_config, mesh, node_info, param_partition, search_alg):
        self.model = model
        self.data_config = data_config
        self.elastic_config = elastic_config
        self.mesh = mesh
        self.node_info = node_info
        self.param_partition = param_partition
        self.search_alg = search_alg
        self.mask_sampler = MaskSampler(data_config, None)

        if elastic_config.scan_layers:
            self._cache_spec = PS(None, ('dp', 'fsdp'), 'sp', 'tp', None)
        else:
            self._cache_spec = PS(('dp', 'fsdp'), 'sp', 'tp', None)

    @cached_property
    def _recon_with_mask(self):
        def fn(params, batch, cache, encoding_mask, cache_idx):
            block_size = self.elastic_config.max_toks
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
            batch['vision'] = batch['vision'].astype(jnp.float32) / 127.5 - 1
            batch_sliced = jax.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                  x, cache_idx, block_size, axis=1),
                batch)
            encoding_mask = with_sharding_constraint(encoding_mask, PS(('dp', 'fsdp'), 'sp'))
            recon, cache = self.model.apply(
                {**params, **cache}, batch_sliced['vision'], encoding_mask,
                batch['attention_mask'], batch['segment_ids'],
                batch_sliced['position_ids'],
                cache_idx, return_stats=False, training=False,
                method=self.model.recon_with_mask,
                mutable=['cache'],
            )
            recon = jnp.clip(recon, -1, 1)
            recon_loss = jnp.mean((recon - batch_sliced['vision']) ** 2, axis=-1)
            recon_loss = jnp.mean(recon_loss, axis=-1)
            return recon_loss, cache
        return pjit(
            fn,
            in_shardings=(
                self.param_partition,
                PS(('dp', 'fsdp'), 'sp'),
                self._cache_spec,
                PS(),
                PS(),
            ),
            out_shardings=(PS(), self._cache_spec),
        )

    @cached_property
    def _recon(self):
        def fn(params, batch, encoding_mask):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
            batch['vision'] = batch['vision'].astype(jnp.float32) / 127.5 - 1
            recon, _, z = self.model.apply(
                params, batch['vision'], encoding_mask,
                batch['attention_mask'], batch['segment_ids'], batch['position_ids'],
                training=False, return_z=True
            )
            recon = jnp.clip(recon, -1, 1)
            recon_loss = jnp.mean((recon - batch['vision']) ** 2, axis=-1)
            recon_loss = recon_loss.reshape(-1, self.elastic_config.max_toks)
            recon_loss = jnp.mean(recon_loss, axis=-1)
            recon = ((recon + 1) * 127.5).astype(jnp.uint8)
            return recon, recon_loss, z
        return pjit(
            fn,
            in_shardings=(self.param_partition, PS(('dp', 'fsdp'), 'sp'), PS()),
            out_shardings=(PS(('dp', 'fsdp'), 'sp'), PS(), PS()),
        )

    @cached_property
    def _gather_dp_sp(self):
        return pjit(
            lambda x: x,
            in_shardings=(PS(('dp', 'fsdp'), 'sp')),
            out_shardings=PS(),
        )

    def _init_cache(self, batch_size):
        seq_len = self.elastic_config.max_sequence_length
        cache_structs = jax.eval_shape(
            partial(self.model.init, method=self.model.recon_with_mask),
            rngs={'params': jax.random.PRNGKey(0), 'sample': jax.random.PRNGKey(0)},
            vision=np.zeros((batch_size, seq_len, np.prod(self.elastic_config.patch_size) * 3), dtype=np.uint8),
            encoding_mask=np.ones((batch_size, seq_len), dtype=bool),
            attention_mask=np.ones((batch_size, seq_len), dtype=bool),
            segment_ids=np.zeros((batch_size, seq_len), dtype=np.int32),
            position_ids=np.zeros((batch_size, seq_len), dtype=np.int32),
            cache_idx=0,
        )['cache']
        sharded_init_fn = pjit(
            lambda: jax.tree_map(jnp.zeros_like, cache_structs),
            in_shardings=(),
            out_shardings=self._cache_spec,
        )
        return {'cache': sharded_init_fn()}

    def inference_linear(self, params, batch, threshold, default_prop_codes=1.0, max_prop_codes=1.0, n_interp=100, log=False):
        B, L = batch['vision'].shape[:2]
        block_size = self.elastic_config.max_toks
        n_blocks = L // block_size
        cache = self._init_cache(batch['vision'].shape[0])

        final_ntoks = []
        all_ntoks = np.exp2(
            np.linspace(np.log2(self.elastic_config.min_toks),
                        np.log2(np.ceil(block_size * max_prop_codes)),
                        n_interp)
        ).astype(int)
        final_encoding_mask = np.zeros(batch['vision'].shape[:2], dtype=bool)
        pbar = tqdm(total=n_blocks * len(all_ntoks), disable=not log)
        for i in range(n_blocks):
            cache_idx = i * block_size
            recon_losses = []
            for j in range(n_interp):
                ntoks = all_ntoks[j]
                encoding_mask = self.mask_sampler(ntoks)[None].repeat(B, axis=0)
                recon_loss, _ = self._recon_with_mask(
                    params, batch, cache, encoding_mask, cache_idx)
                recon_losses.append(recon_loss)
                pbar.update(1)
            recon_losses = np.array(jax.device_get(recon_losses)).T
            is_valid = recon_losses <= threshold
            none_valid = np.all(~is_valid, axis=1)
            idxs = np.argmax(is_valid, axis=-1)
            ntok = all_ntoks[idxs]
            ntok[none_valid] = int(default_prop_codes * self.elastic_config.max_toks)

            encoding_mask = np.zeros((B, block_size), dtype=bool)
            for batch_idx in range(B):
                encoding_mask[batch_idx] = self.mask_sampler(ntok[batch_idx])
                final_encoding_mask[:, i * block_size:(i + 1) * block_size] = encoding_mask
            _, cache = self._recon_with_mask(params, batch, cache, encoding_mask, cache_idx)
            final_ntoks.append(ntok)
        final_ntoks = np.array(final_ntoks).T
        recon, recon_loss, z = self._recon(params, batch, final_encoding_mask)
        z = extract_codes(z, final_encoding_mask, self.elastic_config.max_toks)
        recon_loss = recon_loss.reshape(final_ntoks.shape)
        return recon, recon_loss, final_ntoks, z


    def inference_binary(self, params, batch, threshold, default_prop_codes=1.0, max_prop_codes=1.0, log=False):
        B, L = batch['vision'].shape[:2]
        block_size = self.elastic_config.max_toks
        n_blocks = L // block_size
        cache = self._init_cache(batch['vision'].shape[0])

        final_ntoks = []
        final_encoding_mask = np.zeros((B, L), dtype=bool)
        for i in tqdm(list(range(n_blocks)), disable=not log):
            high = np.full((B,), self.elastic_config.min_toks, dtype=int)
            low = np.full((B,), np.ceil(max_prop_codes * block_size), dtype=int)
            done = np.zeros((B,), dtype=bool)
            cache_idx = i * block_size
            final_ntoks_i = np.zeros((B,), dtype=int)
            recon_losses = []
            count = 0
            while not np.all(done):
                ntok = np.ceil((low + high) / 2)
                encoding_mask = np.zeros((B, block_size), dtype=bool)
                for batch_idx in range(B):
                    encoding_mask[batch_idx] = self.mask_sampler(ntok[batch_idx])
                recon_loss, _ = self._recon_with_mask(
                    params, batch, cache, encoding_mask, cache_idx)
                recon_loss = jax.device_get(recon_loss)
                recon_losses.append(recon_loss)
                is_less = recon_loss < threshold
                low = np.where(is_less & ~done, ntok- 1, low)
                high = np.where(~is_less & ~done, ntok, high)
                final_ntoks_i = np.where(~done, ntok, final_ntoks_i)
                done = done | (low <= high)
                count += 1
            recon_losses = np.array(recon_losses).T
            none_valid = np.all(recon_losses > threshold, axis=1)
            final_ntoks_i[none_valid] = int(default_prop_codes * self.elastic_config.max_toks)
            encoding_mask = np.zeros((B, block_size), dtype=bool)
            for batch_idx in range(B):
                encoding_mask[batch_idx] = self.mask_sampler(final_ntoks_i[batch_idx])
            final_encoding_mask[:, i * block_size:(i + 1) * block_size] = encoding_mask
            _, cache = self._recon_with_mask(params, batch, cache, encoding_mask, cache_idx)
            final_ntoks.append(final_ntoks_i)
        final_ntoks = np.array(final_ntoks).T

        recon, recon_loss, z = self._recon(params, batch, final_encoding_mask)
        z = extract_codes(z, final_encoding_mask, self.elastic_config.max_toks)
        recon_loss = recon_loss.reshape(final_ntoks.shape)
        return recon, recon_loss, final_ntoks, z

    def inference(self, *args, **kwargs):
        if self.search_alg == 'linear':
            fn = self.inference_linear
        elif self.search_alg == 'binary':
            fn = self.inference_binary
        else:
            raise Exception(self.search_alg)
        return fn(*args, **kwargs)

    def save_viz(self, batch, recon, resolution, viz_fname=None):
        recon, vision = self._gather_dp_sp(recon), self._gather_dp_sp(batch['vision'])
        recon = extract_and_reshape(recon, resolution, self.elastic_config)
        vision = extract_and_reshape(vision, resolution, self.elastic_config)
        viz = np.stack([vision, recon], axis=1)
        viz = viz.reshape(-1, *viz.shape[2:])
        path = save_grid(viz, nrow=8, fps=24, fname=viz_fname)
        return path

    def __call__(self, params, batch, threshold, resolution, viz_fname=None):
        recon, _, ntoks, _ = self.inference(params, batch, threshold)
        return self.save_viz(batch, recon, resolution, viz_fname), ntoks
