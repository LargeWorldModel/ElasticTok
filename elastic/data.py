import random
import queue
import os.path as osp
import threading
import io
import multiprocessing as mp
from pathlib import Path
from PIL import Image
from functools import partial

import jax
from jax.experimental.multihost_utils import host_local_array_to_global_array
from jax.sharding import PartitionSpec as PS

import decord
import numpy as np
import gcsfs
from ml_collections import ConfigDict
from tux import open_file


IMAGE_EXT = ['jpg', 'png']
VIDEO_EXT = ['mp4', 'webm', 'avi']


class MaskSampler:
    def __init__(self, config, rng):
        self.mask_type = config.elastic_config.mask_type
        self.min_toks = config.elastic_config.min_toks
        self.max_toks = config.elastic_config.max_toks
        self.rng = rng
        if self.mask_type.startswith('fixed'):
            self.threshold = float(self.mask_type.split('_')[1])
        elif self.mask_type == 'elastic':
            pass
        else:
            raise NotImplementedError(self.mask_type)

    def __call__(self, ntoks=None):
        if ntoks is None:
            if self.mask_type.startswith('fixed'):
                ntoks = int(np.ceil(self.max_toks * self.threshold))
            elif self.mask_type== 'elastic':
                ntoks = self.rng.randint(self.min_toks, self.max_toks)
            else:
                raise NotImplementedError(self.mask_type)
        encoding_mask = np.arange(self.max_toks) <= ntoks
        return encoding_mask


def glob(folder, pattern):
  if folder.startswith('gs://'):
      return [f'gs://{p}' for p in gcsfs.GCSFileSystem().glob(osp.join(folder, pattern))]
  else:
      return list(Path(folder).glob(pattern))


def shard_batch_to_global(batch, mesh):
    if isinstance(batch, dict):
        seq_length = batch[list(batch.keys())[0]].shape[1]
    elif isinstance(batch, (tuple, list)):
        seq_length = batch[0].shape[1]
    else:
        seq_length = batch.shape[1]
    sp_nodes_size = max(1, mesh.shape['sp'] // jax.local_device_count())
    sp_nodes_rank = jax.process_index() % sp_nodes_size
    assert seq_length % sp_nodes_size == 0, (seq_length, sp_nodes_size)
    seq_chunk_size = seq_length // sp_nodes_size
    batch = jax.tree_map(lambda x: x[:, sp_nodes_rank*seq_chunk_size:(sp_nodes_rank+1)*seq_chunk_size], batch)
    batch = host_local_array_to_global_array(batch, mesh, PS(('dp', 'fsdp'), 'sp'))
    return batch


class DatasetFactory(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'vision_dataset'
        config.vision_dataset = VisionDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, **kwargs):
        config = cls.get_default_config(config)
        if config.type == 'vision_dataset':
            return VisionDataset(config.vision_dataset, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {config.type}")

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


def _image_resize_shortest(image, image_size):
    W, H = image.size
    if W > H:
        ratio = image_size / H
        target_W = int(W * ratio)
        if target_W == image_size - 1:
            target_W = image_size
        target_H = image_size
    else:
        ratio = image_size / W
        target_W = image_size
        target_H = int(H * ratio)
        if target_H == image_size - 1:
            target_H = image_size
    image = image.resize((target_W, target_H), Image.BILINEAR)
    return image


def _image_center_crop(image, crop_size):
    W, H = image.size
    left = (W - crop_size[0]) // 2
    top = (H - crop_size[1]) // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return image.crop((left, top, right, bottom))


def _process_image(config, data):
    image_size = config.resolution
    block_size = config.elastic_config.frames_per_block
    Tp, Hp, Wp = config.elastic_config.patch_size
    image = Image.open(io.BytesIO(data)).convert('RGB')
    image = _image_resize_shortest(image, image_size)
    image = _image_center_crop(image, (image_size, image_size))
    video = np.array(image)[None].repeat(block_size, axis=0)
    T, H, W, C = video.shape
    video = video.reshape(T // Tp, Tp, H // Hp, Hp, W // Wp, Wp, C)
    video = np.transpose(video, (0, 2, 4, 1, 3, 5, 6))
    video = video.reshape(-1, np.prod(config.elastic_config.patch_size) * C)
    return [video]


def _video_center_crop_np(video, crop_size):
    T, H, W, _ = video.shape
    target_T, target_H, target_W = crop_size
    start = (T - target_T) // 2
    top = (H - target_H) // 2
    left = (W - target_W) // 2
    end = start + target_T
    bottom = top + target_H
    right = left + target_W
    return video[start:end, top:bottom, left:right, :]


def _process_video(config, data):
    Tp, Hp, Wp = config.elastic_config.patch_size
    block_size = config.elastic_config.frames_per_block
    image_size = config.resolution
    H, W, C = image_size, image_size, 3
    target_fps = config.fps
    try:
        vr = decord.VideoReader(io.BytesIO(data))
    except Exception as e:
        print(f"Error: {e}")
        return [], (0, 0, 0)
    fps = vr.get_avg_fps()
    n_frames = int(len(vr) / fps * target_fps)
    if n_frames < block_size:
        return [], (0, 0, 0)

    idxs = np.linspace(0, len(vr) - 1, n_frames)
    n_blocks = len(idxs) // block_size
    output = []
    for i in range(n_blocks):
        v = vr.get_batch(idxs[i * block_size:(i + 1) * block_size]).asnumpy()
        v = [Image.fromarray(vi) for vi in v]
        v = [_image_resize_shortest(vi, image_size) for vi in v]
        v = [_image_center_crop(vi, (image_size, image_size)) for vi in v]
        v = np.stack([np.array(vi) for vi in v], axis=0)
        v = v.reshape(block_size // Tp, Tp, H // Hp, Hp, W // Wp, Wp, C)
        v = np.transpose(v, (0, 2, 4, 1, 3, 5, 6))
        v = v.reshape(-1, np.prod(config.elastic_config.patch_size) * C)
        output.append(v)
    return output


def _fetch(data_file, config):
    data = open_file(data_file, 'rb').read()
    if any([data_file.endswith(ext) for ext in IMAGE_EXT]):
        data = _process_image(config, data)
    elif any([data_file.endswith(ext) for ext in VIDEO_EXT]):
        data = _process_video(config, data)
    else:
        raise Exception(f"Filetype {data_file} not supported")
    return data


def _prefetch(data_files, node_info, dataset_loc, batch_queue, config):
    start_epoch, start_i, start_j = dataset_loc
    epoch = start_epoch
    fn = partial(_fetch, config=config)
    rng = np.random.RandomState(config.seed + node_info['dp_node_rank'])
    mask_sampler = MaskSampler(config, rng)
    pool = mp.Pool(config.workers)

    batch_size = config.batch_size // node_info['dp_node_size']
    block_size = config.elastic_config.max_toks
    n_blocks = batch_size * config.seq_length // block_size
    def _init_batch():
        return dict(
            vision=np.zeros((n_blocks, block_size, np.prod(config.elastic_config.patch_size) * 3), dtype=np.uint8),
            encoding_mask=np.zeros((n_blocks, block_size), dtype=bool),
            segment_ids=np.zeros((n_blocks, block_size), dtype=np.int32),
            attention_mask=np.ones((n_blocks, block_size), dtype=bool),
            position_ids=np.arange(
              config.seq_length, dtype=np.int32)[None].repeat(
                batch_size, axis=0).reshape(n_blocks, block_size),
        )

    def _chunk_itr(idxs, start_i, start_j_for_first):
        for i, out in enumerate(pool.imap(fn, [data_files[idx] for idx in idxs])):
            for j, o in enumerate(out):
                if i == 0 and j <= start_j_for_first:
                    continue
                yield o, (start_i + i, j)

    batch, batch_idx = None, 0
    while True:
        idxs = list(range(len(data_files)))
        random.Random(config.seed + epoch).shuffle(idxs)
        idxs = np.array_split(idxs, node_info['dp_node_size'])[node_info['dp_node_rank']].tolist()
        for i in range(start_i if epoch == start_epoch else 0, len(idxs), config.chunk_size):
            idxs_chunk = idxs[i:i + config.chunk_size]
            for elem, loc in _chunk_itr(idxs_chunk, i, start_j if epoch == start_epoch and i == start_i else 0):
                if batch is None:
                    batch, batch_idx = _init_batch(), 0
                batch['vision'][batch_idx] = elem
                batch['encoding_mask'][batch_idx] = mask_sampler()
                batch_idx += 1
                if batch_idx >= n_blocks:
                    batch = {k: v.reshape(batch_size, config.seq_length, *v.shape[2:]) for k, v in batch.items()}
                    dataset_loc = (epoch, *loc)
                    batch_queue.put((batch, dataset_loc))
                    batch = None
        epoch += 1


class VisionDataset(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.ext = 'jpg'
        config.batch_size = 4
        config.resolution = 256
        config.fps = 24 # only relevant for video
        config.seq_length = 4096
        config.workers = 32
        config.chunk_size = 64
        config.max_prefetch = 4
        config.seed = 1234
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, node_info, mesh, elastic_config):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self.config.elastic_config = elastic_config
        self.mesh = mesh
        self.node_info = node_info

        self._data_files = list(map(str, glob(config.path, f'**/*.{config.ext}')))
        print(f'Found {len(self._data_files)} data files')
        self._dataset_loc = (0, 0, 0)

    def __iter__(self):
      self.batch_queue = queue.Queue(maxsize=self.config.max_prefetch)
      self.prefetch_thread = threading.Thread(
          target=_prefetch,
          args=(self._data_files, self.node_info, self._dataset_loc, self.batch_queue, self.config))
      self.prefetch_thread.start()
      return self

    def __next__(self):
        batch, self._dataset_loc = self.batch_queue.get()
        batch = shard_batch_to_global(batch, self.mesh)
        return batch

    def get_state_dict(self):
        return dict(
            config=self.config,
            dataset_loc=self._dataset_loc,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._dataset_loc = state_dict.get('dataset_loc', (0, 0, 0))
