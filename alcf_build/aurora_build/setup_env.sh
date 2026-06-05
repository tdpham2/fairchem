#!/bin/bash
# Aurora (Intel XPU) environment setup for FAIRChem
#
# Creates a venv that inherits Intel's XPU-enabled PyTorch from the
# frameworks module, then installs fairchem-core without pulling PyPI torch.
#
# Usage:
#   source alcf_build/aurora_build/setup_env.sh
#
# Why --no-deps?
#   pyproject.toml declares torch~=2.8.0. Intel's XPU PyTorch is a custom
#   build that pip doesn't recognize, so pip downloads upstream (CUDA) torch.
#   --no-deps skips all dependency resolution; we install the rest manually.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 1. Load Intel frameworks (PyTorch + IPEX + oneCCL for XPU)
module load frameworks

# 2. Create venv inheriting system torch
python3 -m venv "$REPO_ROOT/.venv" --system-site-packages

# 3. Activate
source "$REPO_ROOT/.venv/bin/activate"

# 4. Install fairchem-core WITHOUT pulling torch from PyPI
pip install --no-deps -e "$REPO_ROOT/packages/fairchem-core"

# 5. Install non-torch dependencies
pip install \
    "e3nn>=0.5" \
    "numpy>=2.0,<2.5" \
    "scipy>=1.15.0" \
    "lmdb>=1.6.2,<=1.7.3" \
    "numba>=0.62.0" \
    "huggingface_hub>=0.27.1" \
    "ase>=3.26.0" \
    "ase-db-backends>=0.10.0" \
    "monty>=2026.2.18" \
    "clusterscope==0.0.18" \
    setuptools \
    requests \
    orjson \
    tqdm \
    submitit \
    hydra-core \
    torchtnt \
    pyyaml \
    wandb \
    websockets \
    "ray[serve]>=2.53.0"

# 6. Set model cache location (home dirs are small on HPC)
export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$HOME/.cache/fairchem}"

echo ""
echo "FAIRChem environment ready (Aurora XPU)"
echo "  venv:      $REPO_ROOT/.venv"
echo "  torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "  XPU avail: $(python -c 'import torch; print(torch.xpu.is_available())')"
echo "  cache:     $FAIRCHEM_CACHE_DIR"
