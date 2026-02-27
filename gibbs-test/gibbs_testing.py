#!/usr/bin/env python3
"""
Example script for running Gibbs Ensemble Monte Carlo with MLIP-MC.

Gibbs Ensemble MC simulates vapor-liquid phase equilibria using two
cubic simulation boxes that exchange volume and particles at constant
total N, V, and T. This script demonstrates how to set up and run
the MLP_Gibbs class directly.
"""

import os
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.data import vdw_radii

from mlip_mc import MLP_Gibbs, read_gibbs_binary_log


def main() -> None:
    """Run a Gibbs Ensemble MC simulation."""

    # ============================================
    # Configuration - Modify these parameters
    # ============================================

    example_dir = Path(__file__).resolve().parent

    # Molecule species to simulate
    adsorbate_molecule = "CO2"

    # Temperature in Kelvin
    temperature = 250

    # Initial molecule counts in each box
    N1_init = 150  # Box 1 (gas-like, larger box)
    N2_init = 150  # Box 2 (liquid-like, smaller box)

    # Initial cubic box side lengths in Angstrom
    L1_init = 30.0  # Gas box (larger)
    L2_init = 30.0  # Liquid box (smaller)

    # MC simulation steps
    n_equilibration_steps = 1000
    n_production_steps = 1000

    # Model path
    model_path = str(example_dir / "orb-v3-conservative-omol-20250820.ckpt")

    # Device: use 'cuda' for GPU, 'cpu' for CPU
    device = "cpu"

    # Output directory
    output_dir = example_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # Volume move tuning
    max_delta_V = 50.0  # Maximum volume change per move (A^3)

    # ============================================
    # Load model
    # ============================================

    print("Loading MLIP model...")
    from mlip_mc.main import _load_model, _detect_backend

    backend = _detect_backend()
    print(f"  Backend: {backend}")
    model = _load_model(model_path, device=device, backend=backend, orb_model_variant='omol')
    print("  Model loaded.")

    # ============================================
    # Build molecule template
    # ============================================

    atoms_mol = molecule(adsorbate_molecule)

    # ============================================
    # Run Gibbs Ensemble MC
    # ============================================

    print()
    print("=" * 70)
    print("Gibbs Ensemble Monte Carlo Simulation")
    print("=" * 70)
    print(f"  Molecule:       {adsorbate_molecule}")
    print(f"  Temperature:    {temperature} K")
    print(f"  N1 (gas box):   {N1_init}  in  {L1_init:.1f} A box")
    print(f"  N2 (liq box):   {N2_init}  in  {L2_init:.1f} A box")
    print(f"  Equilibration:  {n_equilibration_steps} steps")
    print(f"  Production:     {n_production_steps} steps")
    print(f"  Max delta V:    {max_delta_V} A^3")
    print(f"  Device:         {device}")
    print()

    gibbs = MLP_Gibbs(
        model=model,
        atoms_mol=atoms_mol,
        T=temperature,
        N1_init=N1_init,
        N2_init=N2_init,
        L1_init=L1_init,
        L2_init=L2_init,
        device=device,
        vdw_radii=vdw_radii,
        max_delta_V=max_delta_V,
        n_equilibration_steps=n_equilibration_steps,
        n_production_steps=n_production_steps,
        output_dir=str(output_dir),
        checkpoint_interval=5000,
        write_trajectory=True,
        trajectory_interval=100,
    )

    total_steps = n_equilibration_steps + n_production_steps
    gibbs.run(total_steps)

    # ============================================
    # Display results
    # ============================================

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print(f"  Temperature:    {temperature} K")
    print(f"  Final N1:       {gibbs.N1}")
    print(f"  Final N2:       {gibbs.N2}")
    print(f"  Final V1:       {gibbs.V1:.1f} A^3")
    print(f"  Final V2:       {gibbs.V2:.1f} A^3")

    rho1 = gibbs.N1 / gibbs.V1 if gibbs.V1 > 0 else 0.0
    rho2 = gibbs.N2 / gibbs.V2 if gibbs.V2 > 0 else 0.0
    print(f"  Density box 1:  {rho1:.6f} molecules/A^3")
    print(f"  Density box 2:  {rho2:.6f} molecules/A^3")

    # Read binary log for production-phase averages
    log_path = output_dir / f"log_gibbs_{temperature:.1f}K.bin"
    if log_path.exists():
        data = read_gibbs_binary_log(str(log_path))
        if data:
            # Use the last portion as production data
            prod_data = data[-min(len(data), n_production_steps):]
            avg_rho1 = np.mean([d['rho1'] for d in prod_data])
            avg_rho2 = np.mean([d['rho2'] for d in prod_data])
            print()
            print(f"  Production averages ({len(prod_data)} samples):")
            print(f"    Avg density box 1: {avg_rho1:.6f} molecules/A^3")
            print(f"    Avg density box 2: {avg_rho2:.6f} molecules/A^3")

    print()
    print(f"  Results saved to: {output_dir}")
    print()
    print("=" * 70)
    print("Calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
