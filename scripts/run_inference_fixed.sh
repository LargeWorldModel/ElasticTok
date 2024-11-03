#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export checkpoint_path="gs://rll-tpus-wilson-us-central2/models/elastictok-vae-params"

# NOTE theta=5e6 (different from the FSQ version)

    #--input_path='example/dog.jpg' \
python3 -u scripts/inference_fixed.py \
    --input_path='example/flower.mp4' \
    --output_folder='example/outputs_vae' \
    --n_codes=512 \
    --max_blocks_per_chunk=4 \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_elastic_config='200m' \
    --update_elastic_config="dict(mask_mode='elastic',min_toks=128,max_toks=2048,frames_per_block=4,patch_size=(2,8,8),bottleneck_type='vae',vae_bottleneck_dim=8,theta=5000000,max_sequence_length=8192,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=512,remat_attention='',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='',scan_layers=True)" \
    --load_checkpoint="params::$checkpoint_path"
