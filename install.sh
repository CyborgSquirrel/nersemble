#!/usr/bin/env bash

  # torch==2.0.1+cu117 \

pip install \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  torchvision==0.15.2+cu117 \
  torch \
  einops \
  torch_efficient_distloss \

# Nerfstudio
# pip install nerfstudio==0.3.1
pip install nerfstudio
pip install \
  --find-links https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu117.html \
  nerfacc \
  # nerfacc==0.5.2+pt20cu117 \
# pre-build wheel. Avoids compilation issues

# Custom packages for facilitating ML research
# Misc
pip install \
  dreifus==0.1.2 \
  elias==0.2.3 \
  environs \
  pyfvvdp \
  connected-components-3d \

pip install einops

# tinycudann
# Needs to be installed afterwards
pip install wheel  # https://github.com/NVlabs/tiny-cuda-nn/issues/214#issuecomment-3000257445
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
