import requests
import io
import pickle
import jax.numpy as jnp
import flax.linen as nn


class NetLinLayer(nn.Module):
    features: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dropout(rate=0.5)(x, deterministic=True)
        x = nn.Conv(self.features, (1, 1), padding=0, use_bias=False)(x)
        return x


class VGG16(nn.Module):

    @nn.compact
    def __call__(self, x):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
               'M', 512, 512, 512, 'M', 512, 512, 512]

        layer_ids = [1, 4, 8, 12, 16]
        out = []
        for i, v in enumerate(cfg):
            if v == 'M':
                x = nn.max_pool(x, (2, 2,), strides=(2, 2))
            else:
                x = nn.Conv(v, (3, 3), padding=(1, 1))(x)
                x = nn.relu(x)
                if i in layer_ids:
                    out.append(x)
        return out


def spatial_average(feat, keepdims=True):
    return jnp.mean(feat, axis=[1, 2], keepdims=keepdims)


def normalize(feat, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(feat ** 2, axis=-1, keepdims=True))
    return feat / (norm_factor + eps)


class LPIPS(nn.Module):
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, images_0, images_1):
        shift = jnp.array([-0.030, -0.088, -0.188], dtype=self.dtype)
        scale = jnp.array([0.458, 0.448, 0.450], dtype=self.dtype)
        images_0 = (images_0 - shift) / scale
        images_1 = (images_1 - shift) / scale

        net = VGG16()
        outs_0, outs_1 = net(images_0), net(images_1)
        diffs = []
        for feat_0, feat_1 in zip(outs_0, outs_1):
            diff = (normalize(feat_0) - normalize(feat_1)) ** 2
            diffs.append(diff)

        res = []
        for d in diffs:
            d = NetLinLayer()(d)
            d = spatial_average(d, keepdims=True)
            res.append(d)

        val = sum(res)
        return val


def load_params():
    response = requests.get('https://github.com/wilson1yan/lpips-jax/raw/master/lpips_jax/weights/vgg16.ckpt')
    assert response.status_code == 200
    params = pickle.load(io.BytesIO(response.content))
    return {'params': params}
