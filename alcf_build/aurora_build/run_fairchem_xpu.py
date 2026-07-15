#!/usr/bin/env python
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

FAIRChem geometry optimization on Intel XPU (Aurora HPC), with a CPU
reference run for numerical validation.

Aurora setup:
    module load frameworks   # provides PyTorch + IPEX + oneCCL

    # Pre-download model on login node (compute nodes may lack internet):
    python -c "from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name; pretrained_checkpoint_path_from_name('uma-s-1p1')"

Usage:
    python run_fairchem_xpu.py structure.xyz [model_name] [task_name] [fmax]

    model_name: uma-s-1p1 (default), uma-sm-1p1, etc.
    task_name:  oc20 (default), omat, omol, odac, omc
    fmax:       force convergence criterion in eV/A (default: 0.01)

Runs BFGS on XPU first (writes trajectory + optimized geometry) and then
again on CPU from the original input geometry, and prints both final
energies plus their absolute difference.
"""

from __future__ import annotations

import os
import sys

from ase.io import read, write
from ase.optimize import BFGS

from fairchem.core import FAIRChemCalculator, pretrained_mlip


def run_opt(device, source_atoms, model_name, task_name, fmax, trajectory):
    atoms = source_atoms.copy()
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    atoms.calc = FAIRChemCalculator(predictor, task_name=task_name)

    dyn = BFGS(atoms, trajectory=trajectory)
    dyn.run(fmax=fmax)
    return atoms, dyn.nsteps


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

    source_atoms = read(xyz_path)
    print(f"Loaded {len(source_atoms)} atoms from {xyz_path}")

    out_prefix = os.path.splitext(os.path.basename(xyz_path))[0]
    traj_path = f"{out_prefix}_opt.traj"
    out_path = f"{out_prefix}_optimized.xyz"

    print("\n=== XPU run ===")
    xpu_atoms, xpu_steps = run_opt(
        "xpu", source_atoms, model_name, task_name, fmax, traj_path
    )
    e_xpu = xpu_atoms.get_potential_energy()
    write(out_path, xpu_atoms)

    print("\n=== CPU run (reference) ===")
    cpu_atoms, cpu_steps = run_opt(
        "cpu", source_atoms, model_name, task_name, fmax, None
    )
    e_cpu = cpu_atoms.get_potential_energy()

    print(f"\nOptimization converged (fmax={fmax} eV/A)")
    print(f"XPU final energy:  {e_xpu:.6f} eV  ({xpu_steps} steps)")
    print(f"CPU final energy:  {e_cpu:.6f} eV  ({cpu_steps} steps)")
    print(f"|dE| (XPU-CPU):    {abs(e_xpu - e_cpu):.3e} eV")
    print(f"Trajectory:        {traj_path}  (XPU run)")
    print(f"Optimized geom:    {out_path}  (XPU run)")


if __name__ == "__main__":
    main()
