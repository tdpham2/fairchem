#!/usr/bin/env python
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FAIRChem geometry optimization on Intel XPU (Aurora HPC).

Aurora setup:
    module load frameworks   # provides PyTorch + IPEX + oneCCL

    # Pre-download model on login node (compute nodes may lack internet):
    python -c "from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name; pretrained_checkpoint_path_from_name('uma-s-1p1')"

Usage:
    python run_fairchem_xpu.py structure.xyz [model_name] [task_name] [fmax]

    model_name: uma-s-1p1 (default), uma-sm-1p1, etc.
    task_name:  oc20 (default), omat, omol, odac, omc
    fmax:       force convergence criterion in eV/A (default: 0.01)
"""

from __future__ import annotations

import os
import sys

from ase.io import read, write
from ase.optimize import BFGS

from fairchem.core import FAIRChemCalculator, pretrained_mlip


def main():
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <structure.xyz> " "[model_name] [task_name] [fmax]"
        )
        sys.exit(1)

    xyz_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "uma-s-1p1"
    task_name = sys.argv[3] if len(sys.argv) > 3 else "oc20"
    fmax = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

    # Load structure
    atoms = read(xyz_path)
    print(f"Loaded {len(atoms)} atoms from {xyz_path}")

    # Create FAIRChem calculator on XPU
    predictor = pretrained_mlip.get_predict_unit(model_name, device="xpu")
    calc = FAIRChemCalculator(predictor, task_name=task_name)
    atoms.calc = calc

    # Geometry optimization
    out_prefix = os.path.splitext(os.path.basename(xyz_path))[0]
    traj_path = f"{out_prefix}_opt.traj"
    out_path = f"{out_prefix}_optimized.xyz"

    dyn = BFGS(atoms, trajectory=traj_path)
    dyn.run(fmax=fmax)

    write(out_path, atoms)
    print(f"\nOptimization converged (fmax={fmax} eV/A)")
    print(f"Final energy:    {atoms.get_potential_energy():.6f} eV")
    print(f"Trajectory:      {traj_path}")
    print(f"Optimized geom:  {out_path}")


if __name__ == "__main__":
    main()
