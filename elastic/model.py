from typing import Optional, Tuple, Union
import json
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map
from jax.nn.initializers import variance_scaling
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen import combine_masks

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from ml_collections import ConfigDict
from tux import function_args_to_config, load_pickle, open_file, \
    with_sharding_constraint, get_gradient_checkpoint_policy, get_jax_mesh

from elastic.bottleneck import get_bottleneck



CONFIGS = {
    '200m': {
        'hidden_size': 1024,
        'intermediate_size': 2048,
        'num_encoder_layers': 10,
        'num_decoder_layers': 10,
        'num_attention_heads': 8,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
    },
    'debug': { # A small model for debugging
        'hidden_size': 256,
        'intermediate_size': 256,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'num_attention_heads': 2,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
    },
}


class ElasticTokConfig(PretrainedConfig):
    model_type = "elastic_tok"

    def __init__(
        self,
        mask_type='elastic',
        min_toks=256,
        max_toks=2048,
        frames_per_block=1,
        lpips_loss_ratio=0.1,

        patch_size=(1, 8, 8),
        bottleneck_type='fsq',
        fsq_quant_levels=(8, 8, 8, 5, 5, 5),
        vae_bottleneck_dim=8,
        hidden_size=4096,
        intermediate_size=11008,
        num_encoder_layers=16,
        num_decoder_layers=16,
        num_attention_heads=32,
        max_sequence_length=4096,
        theta=10000,

        rms_norm_eps=1e-5,
        initializer_range=0.02,
        remat_block='',
        remat_attention='',
        remat_mlp='',
        scan_attention=False,
        scan_mlp=False,
        scan_query_chunk_size=1024,
        scan_key_chunk_size=1024,
        scan_mlp_chunk_size=1024,
        scan_layers=True,
        param_scan_axis=0,
        mesh_dim=None,
        use_flash_attention=True,
        **kwargs,
    ):
        self.lpips_loss_ratio = lpips_loss_ratio
        self.mask_type = mask_type
        self.frames_per_block = frames_per_block
        self.min_toks = min_toks
        self.max_toks = max_toks
        self.patch_size = patch_size
        self.bottleneck_type = bottleneck_type
        self.fsq_quant_levels = fsq_quant_levels
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.remat_block = remat_block
        self.remat_attention = remat_attention
        self.remat_mlp = remat_mlp
        self.scan_attention = scan_attention
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.scan_layers = scan_layers
        self.param_scan_axis = param_scan_axis
        self.mesh_dim = mesh_dim
        self.use_flash_attention = use_flash_attention
        self.theta = theta
        super().__init__(**kwargs)

    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'tp', 'sp'))

    @staticmethod
    def get_ranks_and_size(mesh):
        out = dict()
        mp_size = mesh.shape['tp'] * mesh.shape['sp']
        mp_node_size = max(1, mp_size // jax.local_device_count())
        dp_node_size = jax.process_count() // mp_node_size
        out.update(mp_node_size=mp_node_size,
                   dp_node_size=dp_node_size)

        dp_node_rank = jax.process_index() // mp_node_size
        mp_node_rank = jax.process_index() % mp_node_size
        out.update(dp_node_rank=dp_node_rank,
                   mp_node_rank=mp_node_rank)
        return out

    @staticmethod
    def get_partition_rules(scan_layers=False, scan_axis=0):
        if scan_layers:
            if scan_axis == 0:
                return (
                    # encoder layers
                    ("in_proj/kernel", PS(("fsdp", "sp"), "tp")),
                    ("pre_quant/kernel", PS(("fsdp", "sp"), None)),
                    # decoder layers
                    ("post_quant/kernel", PS(None, ("fsdp", "sp"))),
                    ("ln_f/kernel", PS(None)),
                    ("out_proj/kernel", PS(("fsdp", "sp"), "tp")),
                    # attention
                    ("attention/(wq|wk|wv)/kernel", PS(None, ("fsdp", "sp"), "tp")),
                    ("attention/wo/kernel", PS(None, "tp", ("fsdp", "sp"))),
                    # mlp
                    ("feed_forward/w1/kernel", PS(None, "fsdp", "tp")),
                    ("feed_forward/w2/kernel", PS(None, "tp", "fsdp")),
                    ("feed_forward/w3/kernel", PS(None, "fsdp", "tp")),
                    # layer norms
                    ("attention_norm/kernel", PS(None, None)),
                    ("ffn_norm/kernel", PS(None, None)),
                    ('.*', PS(None)),
                )
            else:
                raise ValueError(f"Invalid scan_axis {scan_axis}")
        else:
            return (
                # encoder layers
                ("in_proj/kernel", PS(("fsdp", "sp"), "tp")),
                ("pre_quant/kernel", PS(("fsdp", "sp"), None)),
                # decoder layers
                ("post_quant/kernel", PS(None, ("fsdp", "sp"))),
                ("ln_f/kernel", PS(None)),
                ("out_proj/kernel", PS(("fsdp", "sp"), "tp")),
                # atention
                ("attention/(wq|wk|wv)/kernel", PS(("fsdp", "sp"), "tp")),
                ("attention/wo/kernel", PS("tp", ("fsdp", "sp"))),
                # mlp
                ("feed_forward/w1/kernel", PS(("fsdp", "sp"), "tp")),
                ("feed_forward/w2/kernel", PS("tp", ("fsdp", "sp"))),
                ("feed_forward/w3/kernel", PS(("fsdp", "sp"), "tp")),
                # layer norms
                ("attention_norm/kernel", PS(None)),
                ("ffn_norm/kernel", PS(None)),
                ('.*', PS(None)),
            )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'sample',)

    @classmethod
    def load_config(cls, path):
        if path in CONFIGS:
            return cls.from_dict(CONFIGS[path])
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['llama_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')


remat = nn_partitioning.remat

logger = logging.get_logger(__name__)


class RMSNorm(nn.Module):
    dim: int
    eps: float=1e-6
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


def precompute_freqs_cis(dim, max_position_embedding, theta=10000.0, dtype=jnp.float32):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(max_position_embedding) # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


class Attention(nn.Module):
    config: ElasticTokConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.wq = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            config.max_sequence_length,
            theta=config.theta,
            dtype=self.dtype,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:-1] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:-2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, cache_idx):
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        *batch_dims, _, _, _ = cached_key.value.shape
        indices = (0,) * len(batch_dims) + (cache_idx, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        return key, value

    def __call__(
        self,
        hidden_states,
        attention_mask,
        segment_ids,
        position_ids,
        cache_idx: Optional[int] = None,
    ):
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        if xq.shape[1] == 1:
            xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "tp"))
        else:
            xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), "sp", "tp"))
        xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), "sp", "tp"))
        xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), "sp", "tp"))

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)
        if cache_idx is not None:
            xk, xv = self._concatenate_to_cache(xk, xv, cache_idx)

        if self.config.scan_attention and xq.shape[1] >= max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size):
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

            # transform boolean mask into float mask
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

            q_sp_dim = None if cache_idx is not None else 'sp'
            sp_size = ElasticTokConfig.get_jax_mesh(self.config.mesh_dim).shape['sp']
            assert self.config.max_toks % sp_size == 0 or sp_size % self.config.max_toks == 0
            from ringattention import ringattention
            ring_attention_sharded = shard_map(
                partial(
                    ringattention,
                    axis_name="sp",
                    float32_logits=True,
                    blockwise_kwargs=dict(
                        attn_pdrop=0.,
                        causal_block_size=self.config.max_toks,
                        query_chunk_size=self.config.scan_query_chunk_size,
                        key_chunk_size=self.config.scan_key_chunk_size,
                        dtype=self.dtype,
                        policy=get_gradient_checkpoint_policy('nothing_saveable'),
                        precision=self.precision,
                        prevent_cse=not self.config.scan_layers,
                        deterministic=False,
                        dropout_rng=None,
                    )
                ),
                mesh=ElasticTokConfig.get_jax_mesh(self.config.mesh_dim),
                in_specs=(
                    PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), None, None, None),
                    PS(("dp", "fsdp"), None),
                    PS(),
                ),
                out_specs=PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                check_rep=False
            )
            attn_output = ring_attention_sharded(xq, xk, xv, attention_bias, segment_ids, cache_idx)
            attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), "sp", "tp", None))
        else:
            attention_mask = attention_mask[:, None, None]
            segment_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
            segment_mask = segment_mask[:, None]
            n_blocks = self.config.max_sequence_length // self.config.max_toks
            causal_mask = jnp.tril(jnp.ones((n_blocks, n_blocks), dtype=bool))
            causal_mask = causal_mask.repeat(self.config.max_toks, axis=0).repeat(self.config.max_toks, axis=1)
            causal_mask = causal_mask[None, None]
            attention_mask = combine_masks(attention_mask, segment_mask, causal_mask)
            if cache_idx is not None:
                attention_mask = jax.lax.dynamic_slice_in_dim(attention_mask, cache_idx, xq.shape[1], axis=2)

            q_sp_dim = None if cache_idx is not None else 'sp'
            from ringattention import ringattention_inference
            ring_attention_sharded = shard_map(
                partial(ringattention_inference, axis_name="sp"), mesh=ElasticTokConfig.get_jax_mesh(self.config.mesh_dim),
                in_specs=(
                    PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), "sp", "tp", None),
                    PS(("dp", "fsdp"), None, q_sp_dim, None)
                ),
                out_specs=PS(("dp", "fsdp"), q_sp_dim, "tp", None),
                check_rep=False
            )
            attn_output = ring_attention_sharded(
                xq, xk, xv, attention_mask
            )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        return attn_output


class MLP(nn.Module):
    config: ElasticTokConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        config = self.config

        self.w1 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w2 = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.w3 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        return x


class TransformerBlock(nn.Module):
    config: ElasticTokConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        attention_module = Attention
        mlp_module = MLP
        if self.config.remat_attention != '':
            attention_module = remat(
                attention_module, static_argnums=(4,),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention),
                prevent_cse=not self.config.scan_layers,
            )
        if self.config.remat_mlp != '':
            mlp_module = remat(
                mlp_module,
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp),
                prevent_cse=not self.config.scan_layers,
            )

        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        cache_idx: Optional[int] = None,
    ):
        attn_output = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            segment_ids,
            position_ids,
            cache_idx,
        )
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ffn_norm(hidden_states)
        feed_forward_input = with_sharding_constraint(feed_forward_input, PS(("dp", "fsdp"), "sp", None))

        if self.config.scan_mlp and hidden_states.shape[1] >= self.config.scan_mlp_chunk_size:
            from ringattention import blockwise_feedforward
            feed_forward_hidden_states = blockwise_feedforward(
                self.feed_forward,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
                pre_remat=True
            )
        else:
            feed_forward_hidden_states = self.feed_forward(feed_forward_input)
        feed_forward_hidden_states = with_sharding_constraint(feed_forward_hidden_states, PS(("dp", "fsdp"), None, "tp"))

        hidden_states = hidden_states + feed_forward_hidden_states

        if self.config.scan_layers:
            return (hidden_states, None)
        return hidden_states


class TransformerBlockCollection(nn.Module):
    config: ElasticTokConfig
    layer_key: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        cache_idx=None,
    ):
        block = TransformerBlock
        if self.config.remat_block != '':
            block = remat(
                block, static_argnums=(4,),
                prevent_cse=not self.config.scan_layers,
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        if self.config.scan_layers:
            initializing = self.is_mutable_collection('params')
            params_spec = (
                self.config.param_scan_axis if initializing else
                nn_partitioning.ScanIn(self.config.param_scan_axis))
            cache_spec = 0
            hidden_states, _ = nn.scan(
                block,
                variable_axes={
                    'params': params_spec,
                    'cache': cache_spec,
                    'intermediates': 0
                },
                split_rngs={
                    'params': True,
                },
                in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
                length=getattr(self.config, self.layer_key),
                metadata_params={nn.PARTITION_NAME: 'scan_decoder_layer'},
                )(self.config, name='scan_decoder', dtype=self.dtype, param_dtype=self.param_dtype,)(
                    hidden_states,
                    attention_mask,
                    segment_ids,
                    position_ids,
                    cache_idx,
                )
        else:
            blocks = [
                block(
                    self.config,
                    name=str(i),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ) for i in range(getattr(self.config, self.layer_key))
            ]
            for block in blocks:
                layer_outputs = block(
                    hidden_states,
                    attention_mask,
                    segment_ids,
                    position_ids,
                    cache_idx,
                )
                hidden_states = layer_outputs
        return hidden_states


class ElasticTokEncoder(nn.Module):
    config: ElasticTokConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        kwargs = dict(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.is_kept_embed = self.param(
            'is_kept_embed', variance_scaling(1.0, "fan_in", "normal", out_axis=0),
            (1, self.config.hidden_size,), self.param_dtype
        )
        self.is_masked_embed = self.param(
            'is_masked_embed', variance_scaling(1.0, "fan_in", "normal", out_axis=0),
            (1, self.config.hidden_size,), self.param_dtype
        )
        self.in_proj = nn.Dense(self.config.hidden_size, **kwargs)
        self.encoder_blocks = TransformerBlockCollection(
            self.config, 'num_encoder_layers',
            dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.bottleneck = get_bottleneck(self.config)
        self.ln_f = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)
        self.pre_quant = nn.Dense(self.bottleneck.proj_dim, **kwargs)

    def indexes_to_codes(self, z):
        return self.bottleneck.indexes_to_codes(z)

    def codes_to_indexes(self, z):
        return self.bottleneck.codes_to_indexes(z)

    def __call__(
        self,
        vision,
        encoding_mask,
        attention_mask,
        segment_ids,
        position_ids,
        cache_idx: Optional[int] = None,
        training: bool = True,
    ):
        input_embeds = self.in_proj(vision)
        input_embeds += jnp.where(
            encoding_mask[..., None], self.is_kept_embed,
            self.is_masked_embed)
        hidden_states = self.encoder_blocks(
            input_embeds,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
        )
        hidden_states = self.ln_f(hidden_states)
        z = self.pre_quant(hidden_states)
        return self.bottleneck(
            z, encoding_mask, self.make_rng('sample') if training else None)


class ElasticTokDecoder(nn.Module):
    config: ElasticTokConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        kwargs = dict(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
        )
        self.post_quant = nn.Dense(self.config.hidden_size, **kwargs)
        self.decoder_blocks = TransformerBlockCollection(
            self.config, 'num_decoder_layers',
            dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.ln_f = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)
        self.out_proj = nn.Dense(np.prod(self.config.patch_size) * 3, **kwargs)

    def __call__(
        self,
        z,
        encoding_mask,
        attention_mask,
        segment_ids,
        position_ids,
        cache_idx: Optional[int] = None,
        return_feats=False,
    ):
        hidden_states = self.post_quant(z)
        hidden_states = self.decoder_blocks(
            hidden_states,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
        )
        hidden_states = self.ln_f(hidden_states)
        recon = self.out_proj(hidden_states)
        return jnp.tanh(recon)


class ElasticTok(nn.Module):
    config: ElasticTokConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.encoder = ElasticTokEncoder(
            self.config, self.dtype, self.param_dtype, self.precision
        )
        self.decoder = ElasticTokDecoder(
            self.config, self.dtype, self.param_dtype, self.precision
        )

    def index_to_codes(self, z):
        return self.encoder.indexes_to_codes(z)

    def codes_to_indexes(self, z):
        return self.encoder.codes_to_indexes(z)

    def encode(
        self,
        vision,
        encoding_mask,
        attention_mask,
        segment_ids,
        position_ids,
        cache_idx: Optional[int] = None,
        training: bool = True,
    ):
        z, stats = self.encoder(
            vision,
            encoding_mask,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
            training=training
        )
        z = jnp.where(encoding_mask[..., None], z, 0)
        return z, stats

    def recon_with_mask(
        self,
        vision,
        encoding_mask,
        attention_mask,
        segment_ids,
        position_ids,
        cache_idx: Optional[int] = None,
        return_stats: bool = True,
        training: bool = True,
    ):
        z, stats = self.encode(
            vision,
            encoding_mask,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
            training=training,
        )
        recon = self.decoder(
            z,
            encoding_mask,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
        )
        if return_stats:
            return recon, stats
        else:
            return recon

    def decode(
        self,
        z,
        encoding_mask,
        attention_mask,
        segment_ids,
        position_ids,
        cache_idx: Optional[int] = None,
    ):
        recon  = self.decoder(
            z,
            encoding_mask,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
        )
        return recon

    def __call__(
        self,
        vision,
        encoding_mask,
        attention_mask,
        segment_ids,
        position_ids,
        cache_idx: Optional[int] = None,
        training: bool = True,
        return_z: bool = False,
    ):
        assert vision.shape[1] <= self.config.max_sequence_length
        z, stats = self.encode(
            vision,
            encoding_mask,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
            training=training,
        )
        recon = self.decoder(
            z,
            encoding_mask,
            attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            cache_idx=cache_idx,
        )
        if return_z:
            return recon, stats, self.codes_to_indexes(z)
        else:
            return recon, stats
