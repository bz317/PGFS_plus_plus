#!/usr/bin/env bash
# Install PyTorch (CUDA 12.1) and PyTorch Geometric for PGFS++.
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.3.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.0}"
PYG_VERSION="${PYG_VERSION:-2.7.0}"
CUDA_TAG="${CUDA_TAG:-cu121}"

pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

pip install "torch-geometric==${PYG_VERSION}"

pip install \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"
