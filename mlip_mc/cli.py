#!/usr/bin/env python3
"""
Command-line interface for MLIP-MC.

This module provides the CLI entry point for running GCMC isotherm simulations
and Widom insertion calculations.
"""

import sys
import argparse
import multiprocessing as mp
import traceback
from typing import Union, List
from .main import run_gcmc, run_widom, run_tmmc


def parse_pressures(pressure_str: str) -> Union[float, List[float]]:
    """
    Parse pressure string into float or list of floats.
    
    Parameters
    ----------
    pressure_str : str
        Comma-separated list of pressures or a single number
        
    Returns
    -------
    Union[float, List[float]]
        Single pressure value or list of pressure values
        
    Raises
    ------
    ValueError
        If the pressure string cannot be parsed
    """
    parts = [p.strip() for p in pressure_str.split(',')]
    
    if len(parts) == 1:
        return float(parts[0])
    
    return [float(p) for p in parts]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='MLIP-MC: Monte Carlo Simulations with Machine-Learned Interatomic Potentials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GCMC: Multiple pressure points
  mlip_mc \\
      --mode gcmc \\
      --adsorbent framework.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --pressures 0.1,0.5,1.0,2.0,5.0 \\
      --n-equil 10000 \\
      --n-prod 20000 \\
      --output-dir my_results

  # GCMC: Single pressure point
  mlip_mc \\
      --mode gcmc \\
      --adsorbent framework.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --pressures 1.0 \\
      --n-equil 5000 \\
      --n-prod 15000

  # Widom insertion
  mlip_mc \\
      --mode widom \\
      --adsorbent framework.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --n-trials 10000 \\
      --output-dir widom_results

  # TMMC: full isotherm (with hysteresis) from one simulation,
  # flexible framework via hybrid MC/MD moves
  mlip_mc \\
      --mode tmmc \\
      --adsorbent framework.xyz \\
      --adsorbate-molecule H2O \\
      --temperature 298.0 \\
      --reference-pressure 0.02 \\
      --n-steps 500000 \\
      --n-min 0 --n-max 120 \\
      --md-probability 0.02 \\
      --output-dir tmmc_results
        """
    )

    parser.add_argument('--mode', type=str, default='gcmc', choices=['gcmc', 'widom', 'tmmc'],
                        help='Simulation mode: gcmc (Grand Canonical Monte Carlo), widom (Widom insertion), '
                             'or tmmc (Transition-Matrix Monte Carlo) (default: gcmc)')
    parser.add_argument('--adsorbent', type=str, required=True,
                        help='Path to adsorbent structure file (.xyz, .cif, etc.)')
    parser.add_argument('--adsorbate-path', type=str, default=None,
                        help='Path to adsorbate structure file (.xyz, .cif, etc.)')
    parser.add_argument('--adsorbate-molecule', type=str, default=None,
                        help='Name of adsorbate molecule (e.g., CO2, CH4) if not using file')
    parser.add_argument('--temperature', type=float, required=True,
                        help='Temperature in Kelvin (required)')
    parser.add_argument('--pressures', type=str, default=None,
                        help='Comma-separated list of pressures in bar, or single number (required for GCMC mode)')
    parser.add_argument('--n-equil', type=int, default=10000,
                        help='Number of equilibration steps for GCMC (default: 10000)')
    parser.add_argument('--n-prod', type=int, default=20000,
                        help='Number of production steps for GCMC (default: 20000)')
    parser.add_argument('--n-trials', type=int, default=10000,
                        help='Number of Widom insertion trials (default: 10000)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to MLIP model file (required). Can be a local path or Hugging Face URI (e.g., hf://your-org/your-repo). '
                             'For metatomic, pass the TorchScript model exported by metatrain (mtt export).')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['fairchem', 'mace-torch', 'orb-models', 'metatomic'],
                        help='MLIP backend to use. Default: auto-detect from installed packages '
                             '(in the order fairchem, mace-torch, orb-models, metatomic). Set explicitly '
                             'when several backends are installed, e.g. --backend metatomic to run a '
                             'metatrain model in an image that also ships MACE.')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='Hugging Face authentication token (optional, uses cached token if available)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID to use (for Widom mode, default: 0, use -1 for CPU)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='Interval for saving checkpoints (default: 10000)')
    parser.add_argument('--write-trajectory', action='store_true',
                        help='Write trajectory files')
    parser.add_argument('--trajectory-interval', type=int, default=100,
                        help='Interval for writing structures (default: 100)')
    parser.add_argument('--overwrite-checkpoints', action='store_true',
                        help='Write trajectory files')
    parser.add_argument('--orb-variant', type=str, default='omat', choices=['omat', 'omol'],
                        help='ORB model variant: omat (default, orb_v3_conservative_inf_omat) or omol (orb_v3_conservative_omol, charge=0 spin=1)')

    # TMMC-specific options
    tmmc_group = parser.add_argument_group('TMMC options (--mode tmmc)')
    tmmc_group.add_argument('--reference-pressure', type=float, default=1.0,
                            help='Reference pressure in bar at which ln Pi is sampled (default: 1.0)')
    tmmc_group.add_argument('--n-steps', type=int, default=100000,
                            help='Total number of TMMC steps (default: 100000)')
    tmmc_group.add_argument('--n-min', type=int, default=0,
                            help='Lower bound of the macrostate window (default: 0)')
    tmmc_group.add_argument('--n-max', type=int, default=100,
                            help='Upper bound of the macrostate window (default: 100)')
    tmmc_group.add_argument('--bias-update-interval', type=int, default=10000,
                            help='Steps between ln Pi bias refreshes (default: 10000)')
    tmmc_group.add_argument('--md-probability', type=float, default=0.0,
                            help='Probability of hybrid MC/MD framework-flexibility moves (default: 0.0 = rigid)')
    tmmc_group.add_argument('--md-ensemble', type=str, default='nvt', choices=['nvt', 'npt'],
                            help='Ensemble for hybrid MD moves (default: nvt)')
    tmmc_group.add_argument('--md-steps', type=int, default=1000,
                            help='MD steps per hybrid move (default: 1000)')
    tmmc_group.add_argument('--md-timestep', type=float, default=1.0,
                            help='MD timestep in fs (default: 1.0)')
    tmmc_group.add_argument('--volume-probability', type=float, default=0.0,
                            help='Probability of MC volume moves (default: 0.0)')
    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    args = parse_arguments()
    
    # Validate arguments
    if not args.adsorbate_path and not args.adsorbate_molecule:
        print("ERROR: Either --adsorbate-path or --adsorbate-molecule must be provided", 
              file=sys.stderr)
        sys.exit(1)
    
    # Handle GPU ID for Widom mode
    gpu_id = args.gpu_id if args.gpu_id >= 0 else 'cpu'
    
    # Run simulation based on mode
    try:
        if args.mode == 'tmmc':
            # Transition-Matrix Monte Carlo mode
            isotherm_pressures = None
            if args.pressures is not None:
                pressures = parse_pressures(args.pressures)
                isotherm_pressures = pressures if isinstance(pressures, list) else [pressures]

            run_tmmc(
                adsorbent_path=args.adsorbent,
                adsorbate_path=args.adsorbate_path,
                adsorbate_molecule=args.adsorbate_molecule,
                temperature=args.temperature,
                reference_pressure=args.reference_pressure,
                n_steps=args.n_steps,
                N_min=args.n_min,
                N_max=args.n_max,
                bias_update_interval=args.bias_update_interval,
                isotherm_pressures=isotherm_pressures,
                md_probability=args.md_probability,
                md_ensemble=args.md_ensemble,
                md_steps=args.md_steps,
                md_timestep=args.md_timestep,
                volume_probability=args.volume_probability,
                model_path=args.model,
                output_dir=args.output_dir,
                hf_token=args.hf_token,
                gpu_id=gpu_id,
                orb_model_variant=args.orb_variant,
                backend=args.backend,
            )
        elif args.mode == 'widom':
            # Widom insertion mode
            if args.pressures is not None:
                print("WARNING: --pressures is ignored in Widom mode", file=sys.stderr)
            
            run_widom(
                adsorbent_path=args.adsorbent,
                adsorbate_path=args.adsorbate_path,
                adsorbate_molecule=args.adsorbate_molecule,
                temperature=args.temperature,
                n_trials=args.n_trials,
                model_path=args.model,
                output_dir=args.output_dir,
                hf_token=args.hf_token,
                gpu_id=gpu_id,
                orb_model_variant=args.orb_variant,
                backend=args.backend,
            )
        else:
            # GCMC mode
            if args.pressures is None:
                print("ERROR: --pressures is required for GCMC mode", file=sys.stderr)
                sys.exit(1)
            
            # Parse pressure points
            try:
                pressure_points = parse_pressures(args.pressures)
            except ValueError as e:
                print(f"ERROR: Invalid pressure format: {args.pressures}", file=sys.stderr)
                print("Please provide comma-separated numbers or a single number", file=sys.stderr)
                sys.exit(1)
            
            run_gcmc(
                adsorbent_path=args.adsorbent,
                adsorbate_path=args.adsorbate_path,
                adsorbate_molecule=args.adsorbate_molecule,
                temperature=args.temperature,
                pressure_points=pressure_points,
                n_equilibration_steps=args.n_equil,
                n_production_steps=args.n_prod,
                model_path=args.model,
                output_dir=args.output_dir,
                hf_token=args.hf_token,
                checkpoint_interval=args.checkpoint_interval,
                write_trajectory=args.write_trajectory,
                trajectory_interval=args.trajectory_interval,
                overwrite_checkpoints=args.overwrite_checkpoints,
                orb_model_variant=args.orb_variant,
                backend=args.backend,
            )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

