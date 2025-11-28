#!/usr/bin/env python3
"""
MLIP-MC: Monte Carlo Simulations with Machine-Learned Interatomic Potentials

Standalone entry point for running GCMC isotherm simulations.
This script imports from the mlip_mc package to avoid code duplication.
"""

import sys
import os
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import the main function from the package
try:
    from mlip_mc.main import main
except ImportError as e:
    print("ERROR: Could not import mlip_mc package.")
    print(f"Import error: {e}")
    print("\nPlease either:")
    print("  1. Install the package: pip install -e .")
    print("  2. Or use the installed command: mlip_mc")
    sys.exit(1)


if __name__ == "__main__":
    main()
