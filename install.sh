#!/usr/bin/env bash

set -x

  # torch==2.0.1+cu117 \

# pip install \
#   --extra-index-url https://download.pytorch.org/whl/cu117 \
#   torchvision==0.15.2+cu117 \
#   torch \

pip install \
  einops \
  torch_efficient_distloss \

# NerfAcc
# NOTE(andrei): Use pre-built wheel to avoid having to compile CUDA stuff.
pip install --find-links https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu117.html \
  nerfacc \

# Nerfstudio
# pip install nerfstudio==0.3.1
pip install git+https://github.com/CyborgSquirrel/nerfstudio@v0.3.1-fixed

# Custom packages for facilitating ML research
# Misc
pip install \
  dreifus==0.1.2 \
  elias==0.2.3 \
  environs \
  pyfvvdp \
  connected-components-3d \

pip install wheel  # https://github.com/NVlabs/tiny-cuda-nn/issues/214#issuecomment-3000257445

# tinycudann
# NOTE(andrei): Harcoded pre-built tinycudann.
pip install ./tinycudann-2.0-cp312-cp312-linux_x86_64.whl

# pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install .
