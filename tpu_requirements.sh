#! /bin/bash

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    ffmpeg

# Update pip
pip install --upgrade pip

# Python dependencies
cat > $HOME/tpu_requirements.txt <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.4.34
flax==0.10.1
optax==0.2.3
einops
tqdm
ml_collections
wandb
gcsfs
tux @ git+https://github.com/haoliuhl/tux.git
Pillow
ffmpeg-python
decord
ipdb
ringattention
EndOfFile

pip install --upgrade -r $HOME/tpu_requirements.txt
