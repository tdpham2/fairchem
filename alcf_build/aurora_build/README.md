# FAIRChem on Aurora (Intel XPU)

Scripts for running FAIRChem (UMA) geometry optimizations on ALCF Aurora using Intel Data Center Max GPUs (XPU).

## Setup

Aurora's `frameworks` module provides Intel's XPU-enabled PyTorch (IPEX). FAIRChem's `pyproject.toml` declares `torch~=2.8.0` which causes `pip install` to pull the upstream CUDA torch from PyPI, clobbering the Intel build. The setup script avoids this:

```bash
source alcf_build/aurora_build/setup_env.sh
```

This will:
1. Load the `frameworks` module (PyTorch + IPEX + oneCCL)
2. Create a venv with `--system-site-packages` (inherits Intel torch)
3. Install fairchem-core with `pip install --no-deps -e packages/fairchem-core` (skips PyPI torch)
4. Install all non-torch dependencies separately

### Model cache

FAIRChem downloads UMA model checkpoints from HuggingFace to `~/.cache/fairchem/`. This is problematic on HPC (small home dirs, compute nodes without internet).

Pre-download models on a login node before submitting jobs:

```bash
python -c "from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name; pretrained_checkpoint_path_from_name('uma-s-1p1')"
```

## Single-tile geometry optimization

```bash
python alcf_build/aurora_build/run_fairchem_xpu.py structure.xyz [model_name] [task_name] [fmax]
```

- `model_name`: `uma-s-1p1` (default), `uma-sm-1p1`, etc.
- `task_name`: `oc20` (default), `omat`, `omol`, `odac`, `omc`
- `fmax`: force convergence in eV/A (default: `0.01`)
- Output: `{name}_opt.traj` (trajectory) and `{name}_optimized.xyz`

## Multi-tile batch geometry optimization

Distributes structures across all 12 XPU tiles (6 GPUs x 2 tiles) using MPI:

```bash
mpiexec -n 12 --ppn 12 \
    --cpu-bind list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79 \
    python alcf_build/aurora_build/run_fairchem_xpu_batch.py structures.xyz [model_name] [task_name] [fmax]
```

Each MPI rank is pinned to a separate tile via `ZE_AFFINITY_MASK` (set automatically from `PALS_LOCAL_RANKID`). Structures are distributed round-robin across ranks.

If a single structure is provided, all ranks compute it (verification mode) to confirm all tiles are working.

### PBS job submission

```bash
qsub alcf_build/aurora_build/submit_batch.sh -- structures.xyz uma-s-1p1 oc20 0.01
```

Edit `submit_batch.sh` to set your project allocation (`-A`).

## Files

| File | Description |
|------|-------------|
| `setup_env.sh` | Environment setup (venv + deps without PyPI torch) |
| `run_fairchem_xpu.py` | Single-tile geometry optimization |
| `run_fairchem_xpu_batch.py` | Multi-tile batch geometry optimization (MPI) |
| `submit_batch.sh` | PBS job script with CPU binding for Aurora |

## Known issues

- **XPU not auto-detected by upstream FAIRChem**: The upstream codebase only checks for CUDA. This branch adds `device="xpu"` support throughout. You must pass `device="xpu"` explicitly or use this patched version which auto-detects XPU.
- **`torch.compile` on XPU**: IPEX has partial `torch.compile` support. The default inference settings (`execution_mode=None`) will fall back to the general PyTorch backend, which works on XPU. The `umas_fast_gpu` execution mode (Triton kernels) is CUDA-only and is automatically skipped on XPU.
- **CUDA Graphs**: Disabled automatically on XPU. The `wigner_cuda` optimization is CUDA-only and will not be used.
- **float32 non-determinism**: Energies may differ by ~5 uV/atom across tiles due to GPU floating-point reduction order. This is normal.
- **ASE MPI-aware I/O**: When `mpi4py` is imported, ASE restricts file writes to rank 0. The batch script gathers all optimized structures to rank 0 and writes using `write_extxyz` with Python's built-in `open()` to bypass this.
