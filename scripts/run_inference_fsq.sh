#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export checkpoint_path=""

# NOTE theta=5e7 (different from the VAE version)

    #--input_path='example/dog.jpg' \
python3 -u scripts/inference.py \
    --input_path='example/flower.mp4' \
    --output_folder='example/outputs_fsq' \
    --max_blocks_per_chunk=4 \
    --threshold=0.003 \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_elastic_config='200m' \
    --update_elastic_config="dict(mask_mode='elastic',min_toks=256,max_toks=4096,frames_per_block=4,patch_size=(1,8,8),bottleneck_type='fsq',fsq_quant_levels=(8,8,8,5,5,5),theta=50000000,max_sequence_length=16384,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=512,remat_attention='',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='',scan_layers=True)" \
    --load_checkpoint="params::$checkpoint_path"
