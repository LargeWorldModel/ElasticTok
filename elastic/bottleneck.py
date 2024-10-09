import numpy as np
import jax
import jax.numpy as jnp


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)

    def sample(self, rng):
        x = self.mean + self.std * jax.random.normal(rng, self.mean.shape, dtype=self.mean.dtype)
        return x

    def kl(self, mask, block_size):
        ndim = self.mean.ndim
        assert ndim == 3, self.mean.shape
        B, L = self.mean.shape[:2]
        n_blocks = L // block_size
        kl = self.mean ** 2 + self.var - 1.0 - self.logvar
        mask = mask.reshape(B, n_blocks, block_size)
        kl = kl.reshape(B, n_blocks, block_size, kl.shape[-1])
        kl = 0.5 * jnp.sum(kl * mask[..., None], axis=(2, 3))
        kl = jnp.mean(kl, axis=1)
        return kl

    def nll(self, sample, axis=None):
        raise NotImplementedError
        ndim = self.mean.ndim
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * jnp.sum(
            logtwopi + self.logvar + (sample - self.mean) ** 2 / self.var,
            axis=list(range(ndim))
        )

    def mode(self):
        return self.mean


class FSQ:
    def __init__(self, levels):
        self._levels = levels
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(
            ([1], np.cumprod(self._levels_np[:-1]))
        ).astype(np.uint32)

    @property
    def n_codes(self):
        return np.prod(self._levels)

    @property
    def proj_dim(self):
        return len(self._levels)

    def _round_ste(self, z):
        z_hat = jnp.round(z)
        return z + jax.lax.stop_gradient(z_hat - z)

    def bound(self, z):
        eps = 1e-3
        half_l = (self._levels_np - 1) * (1 - eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        z = self.bound(z)
        quantized = self._round_ste(z)
        half_width = self._levels_np // 2
        return quantized / half_width

    def __call__(self, z, encoding_mask, rng):
        z_st = self.quantize(z)
        code_idxs = self.codes_to_indexes(z_st)
        code_idxs = code_idxs.reshape(-1)
        codes_one_hot = jax.nn.one_hot(code_idxs, self.n_codes)
        avg_probs = jnp.mean(codes_one_hot, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
        usage = jnp.mean(jnp.sum(codes_one_hot, axis=0) >= 1)
        return z_st, dict(perplexity=perplexity, codebook_usage=usage)

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels_np // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat):
        assert zhat.shape[-1] == len(self._levels)
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype(jnp.uint32)

    def indexes_to_codes(self, indices):
        indices = indices[..., jnp.newaxis]
        codes_non_centered = jnp.mod(
            jnp.floor_divide(indices, self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(codes_non_centered)


class VAE:
    def __init__(self, bottleneck_dim):
        self.bottleneck_dim = bottleneck_dim

    @property
    def n_codes(self):
        raise NotImplementedError

    @property
    def proj_dim(self):
        return 2 * self.bottleneck_dim

    def __call__(self, z, encoding_mask, rng):
        z = DiagonalGaussianDistribution(z)
        posterior = z
        if rng is None:
            z = posterior.mode()
        else:
            z = posterior.sample(rng)
        kl_loss = posterior.kl(encoding_mask, 2048) # TODO hardocded
        kl_loss = 1e-8 * jnp.sum(kl_loss) / kl_loss.shape[0]
        return z, dict(aux_loss=kl_loss)

    def codes_to_indexes(self, z):
        return z

    def indexes_to_codes(self, z):
        return z


def get_bottleneck(config):
    if config.bottleneck_type == 'fsq':
        return FSQ(config.fsq_quant_levels)
    elif config.bottleneck_type == 'vae':
        return VAE(config.vae_bottleneck_dim)
    else:
        raise ValueError(f'Unknown bottleneck_type: {config.bottleneck_type}')
