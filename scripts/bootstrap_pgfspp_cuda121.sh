#!/usr/bin/env bash
# Install PyTorch (CUDA 12.1) and, when possible, PyTorch Geometric for PGFS++.
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.3.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.0}"
PYG_VERSION="${PYG_VERSION:-2.7.0}"
CUDA_TAG="${CUDA_TAG:-cu121}"

pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ΔSEH scoring needs torch-geometric + torch-sparse. ΔQED training/eval does not.
if [[ "${SKIP_PYG:-0}" == "1" ]]; then
  echo "SKIP_PYG=1: not installing PyTorch Geometric (ΔQED-only setup)."
  exit 0
fi

set +e
pip install "torch-geometric==${PYG_VERSION}"
pip install \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
pyg_status=$?
set -e

if [[ "${pyg_status}" -ne 0 ]]; then
  echo "warning: PyTorch Geometric extras failed to install." >&2
  echo "ΔQED training and eval still work. ΔSEH requires a working PyG stack." >&2
  exit 0
fi

python -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"
python -c "import torch_sparse; print('torch_sparse ok')"
