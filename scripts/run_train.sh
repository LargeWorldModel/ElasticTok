#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export project_id='elastic'
export experiment_id='example-train'
export output_dir="logs"

export WANDB_API_KEY='' # Set if logger.online=True
export dataset_path="" # Supports an arbitrary nested folder of images or videos. You can downoad and extract UCF-101 as a toy example. Point to the UCF-101 folder
export dataset_ext="" # Extension of images or videos (png, jpg, mp4, avi, webm, etc.). For UCF-101 set to "avi"

# The example trains on blocks of 'frames_per_block=4, 256 x 256' videos. Given a `patch_size=(2,8,8)`, each 4x256x256 block is 2048 tokens.
# This means that seq_length=4096 is 2 blocks, or 8 frames of video.

python3 -u scripts/train.py \
    --mesh_dim='!-1,1,1,1' \
    --dtype='fp32' \
    --total_steps=2000000 \
    --log_freq=100 \
    --eval_freq=5000 \
    --eval_thresholds='0.003' \
    --save_model_freq=0 \
    --save_milestone_freq=5000 \
    --load_elastic_config='200m' \
    --update_elastic_config="dict(lpips_loss_ratio=0.1,mask_mode='elastic',min_toks=128,frames_per_block=4,max_toks=2048,patch_size=(2,8,8),bottleneck_type='vae',vae_bottleneck_dim=8,theta=5000000,max_sequence_length=4096,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=512,remat_attention='',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='',scan_layers=True)" \
    --load_checkpoint='' \
    --load_dataset_state='' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=1e-4 \
    --optimizer.adamw_optimizer.lr=1e-4 \
    --optimizer.adamw_optimizer.end_lr=1e-4 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=2000000 \
    --train_dataset.type='vision_dataset' \
    --train_dataset.vision_dataset.path="$dataset_path" \
    --train_dataset.vision_dataset.ext="$dataset_ext" \
    --train_dataset.vision_dataset.batch_size=64 \
    --train_dataset.vision_dataset.seq_length=4096 \
    --train_dataset.vision_dataset.resolution=256 \
    --train_dataset.vision_dataset.fps=4 \
    --train_dataset.vision_dataset.workers=32 \
    --train_dataset.vision_dataset.chunk_size=8 \
    --train_dataset.vision_dataset.max_prefetch=4 \
    --train_dataset.vision_dataset.seed=1234 \
    --checkpointer.save_optimizer_state=True \
    --autoresume=False \
    --logger.append_uuid=False \
    --logger.online=False \
    --logger.project_id="$project_id" \
    --logger.experiment_id="$experiment_id" \
    --logger.output_dir="$output_dir/$project_id" \
    --logger.wandb_dir="$output_dir/$project_id"
