import sys
import os.path as osp
from tux import StreamingCheckpointer


path = sys.argv[1]
path = osp.expanduser(path)
trainstate_path = osp.join(path, 'streaming_train_state')
print('Converting', trainstate_path)
_, params = StreamingCheckpointer.load_trainstate_checkpoint(f'trainstate_params::{trainstate_path}')
params_path = osp.join(path, 'params')
StreamingCheckpointer.save_train_state_to_file(params['params'], params_path, float_dtype='bf16')
print('Saved to', params_path)
