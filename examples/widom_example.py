#!/usr/bin/env python3
"""
Example script for running Widom insertion calculations with MLIP-MC.

Widom insertion is a method to calculate Henry's constants and adsorption
energies by performing test insertions of adsorbate molecules into the
framework at random positions and orientations.

This script demonstrates how to use the simplified run_widom() function
which handles all the setup (model loading, structure loading, etc.)
automatically.
"""

from pathlib import Path

from mlip_mc import run_widom


def main() -> None:
    """Run a Widom insertion calculation."""
    
    # ============================================
    # Configuration - Modify these parameters
    # ============================================
    
    # Framework structure file (relative to this script or absolute path)
    example_dir = Path(__file__).resolve().parent
    framework_path = example_dir / "ZIF8.xyz"
    
    # Adsorbate: either provide a path or molecule name
    adsorbate_path = None  # e.g., "path/to/adsorbate.xyz"
    adsorbate_molecule = "CO2"  # e.g., "CO2", "CH4", "H2O", etc.
    
    # Temperature in Kelvin
    temperature = 298.0
    
    # Number of Widom insertion trials
    n_trials = 10000
    
    # Model path — provide your own local path or HuggingFace URI (hf://your-org/your-repo)
    model_path = "models/model.pt"  # CHANGE THIS to your model path
    
    # Output directory
    output_dir = example_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # ============================================
    # Run Widom insertion
    # ============================================
    
    print("Running Widom insertion calculation...")
    print(f"  Framework: {framework_path}")
    print(f"  Adsorbate: {adsorbate_molecule}")
    print(f"  Temperature: {temperature} K")
    print(f"  Trials: {n_trials}")
    print()
    
    # Run the simulation - this handles everything automatically!
    results = run_widom(
        adsorbent_path=str(framework_path),
        adsorbate_path=str(adsorbate_path) if adsorbate_path else None,
        adsorbate_molecule=adsorbate_molecule,
        temperature=temperature,
        n_trials=n_trials,
        model_path=model_path,
        output_dir=str(output_dir)
    )
    
    # ============================================
    # Display results
    # ============================================
    
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    
    print(f"Temperature: {results.get('temperature', temperature)} K")
    print(f"Total attempts: {results.get('attempts', 0)}")
    print(f"Valid insertions: {results.get('valid_insertions', 0)}")
    print(f"VDW overlaps: {results.get('vdw_overlaps', 0)}")
    print()
    
    if 'widom_adsorption_energy' in results:
        print(f"Widom adsorption energy: {results['widom_adsorption_energy']:.5f} eV")
        print(f"Arithmetic average energy: {results['arithmetic_adsorption_energy']:.5f} eV")
        print(f"Average Boltzmann factor: {results['average_boltzmann_factor']:.5e}")
    
    print()
    print(f"Results saved to: {output_dir}/widom_results.json")
    print(f"Checkpoints saved to: {output_dir}/checkpoints_widom/")
    print()
    print("=" * 70)
    print("Calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

