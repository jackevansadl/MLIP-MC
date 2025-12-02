#!/usr/bin/env python3
"""
Command-line interface for MLIP-MC.

This module provides the CLI entry point for running GCMC isotherm simulations.
"""

import sys
import argparse
import multiprocessing as mp
import traceback
from typing import Union, List
from .main import run_gcmc


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
  # Multiple pressure points
  mlip_mc \\
      --adsorbent framework.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --pressures 0.1,0.5,1.0,2.0,5.0 \\
      --n-equil 10000 \\
      --n-prod 20000 \\
      --output-dir my_results

  # Single pressure point
  mlip_mc \\
      --adsorbent framework.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --pressures 1.0 \\
      --n-equil 5000 \\
      --n-prod 15000

  # Skip plotting
  mlip_mc \\
      --adsorbent framework.xyz \\
      --adsorbate-molecule CO2 \\
      --temperature 298.0 \\
      --pressures 1.0 \\
      --no-plot
        """
    )
    
    parser.add_argument('--adsorbent', type=str, required=True,
                        help='Path to adsorbent structure file (.xyz, .cif, etc.)')
    parser.add_argument('--adsorbate-path', type=str, default=None,
                        help='Path to adsorbate structure file (.xyz, .cif, etc.)')
    parser.add_argument('--adsorbate-molecule', type=str, default=None,
                        help='Name of adsorbate molecule (e.g., CO2, CH4) if not using file')
    parser.add_argument('--temperature', type=float, required=True,
                        help='Temperature in Kelvin (required)')
    parser.add_argument('--pressures', type=str, required=True,
                        help='Comma-separated list of pressures in bar, or single number (required)')
    parser.add_argument('--n-equil', type=int, default=10000,
                        help='Number of equilibration steps (default: 10000)')
    parser.add_argument('--n-prod', type=int, default=20000,
                        help='Number of production steps (default: 20000)')
    parser.add_argument('--model', type=str, default='models/model.pt',
                        help='Path to MLIP model file. Can be a local path or Hugging Face repo (e.g., fengxuyoung/MLIP-MC). Missing files auto-download and are cached.')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='Hugging Face authentication token (optional, uses cached token if available)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='Interval for saving checkpoints (default: 1000)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating isotherm plot')
    
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
    
    # Parse pressure points
    try:
        pressure_points = parse_pressures(args.pressures)
    except ValueError as e:
        print(f"ERROR: Invalid pressure format: {args.pressures}", file=sys.stderr)
        print("Please provide comma-separated numbers or a single number", file=sys.stderr)
        sys.exit(1)
    
    # Run simulation
    try:
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
            plot_isotherm=not args.no_plot,
            hf_token=args.hf_token,
            save_interval=args.save_interval
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

