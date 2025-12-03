# MLIP-MC

ASE framework for Monte Carlo simulations with universal Machine-Learned Interatomic Potentials (MLIP).

## Overview

MLIP-MC is a Python package for performing Monte Carlo simulations of gas adsorption in porous materials using machine-learned interatomic potentials. The package integrates seamlessly with the ASE (Atomic Simulation Environment) framework and supports MLIP models from both **FAIRChem** and **MACE-Torch** backends. This branch supports the molmod models that this software was based on.

## Installation

### Backend Selection

MLIP-MC (molmod branch) supports nequip backends. You must install:

- **nequip**: For molmod trained models

### Quick Install

Install MLIP-MC:

**With nequip backend:**
```bash
# rocm install: pip install torch==1.12.1+rocm5.1.1 torchvision==0.13.1+rocm5.1.1 torchaudio==0.12.1+rocm5.1.1 --extra-index-url https://download.pytorch.org/whl/rocm5.1.1 numpy==1.26.4 Cython==0.29.37
pip install .
```

## Usage

```bash
# multiple pressure points (auto-distributed across GPUs)
mlip_mc \\
    --adsorbent framework.xyz \\
    --adsorbate-molecule CO2 \\
    --temperature 298.0 \\
    --pressures 0.1,0.5,1.0,2.0,5.0,10.0,20.0 \\
    --n-equil 10000 \\
    --n-prod 20000 \\
    --save-interval 1000 \\
    --output-dir results
```

**Command-Line Arguments:**
- `--adsorbent`: Path to adsorbent structure file (.xyz, .cif, etc.) **(required)**
- `--adsorbate-path`: Path to adsorbate structure file (optional). The chemical formula will be automatically extracted to match with the fugacity table.
- `--adsorbate-molecule`: Molecule name (e.g., CO2, CH4) if not using file. This name will be used to match with the fugacity table.
- `--temperature`: Temperature in Kelvin **(required)**
- `--pressures`: Comma-separated pressures in bar, or single number **(required)**
- `--n-equil`: Number of equilibration steps (default: 10000)
- `--n-prod`: Number of production steps (default: 20000)
- `--save-interval`: Interval for saving history checkpoints (default: 1000)
- `--model`: Path to MLIP model file. Can be a local path (default: `models/model.pt`) or a Hugging Face repository name like `fengxuyoung/MLIP-MC` (or `hf://fengxuyoung/MLIP-MC`). Missing files are automatically downloaded and cached. The model format should match your installed backend (nequip `.pth` files).
- `--output-dir`: Output directory (default: results)
- `--hf-token`: Hugging Face access token for downloading private models or bypassing interactive login

**Model caching:** Hugging Face downloads are cached under `~/.cache/mlip-mc/<repo>/<filename>` (or a custom directory set via the `MLIP_MC_CACHE` environment variable). Subsequent runs reuse the cached file even when launched from different working directories.

**Note:** The adsorbate name for EOS (fugacity) calculation is automatically determined:
- If `--adsorbate-molecule` is provided, that name is used to match with the fugacity table
- If `--adsorbate-path` is provided, the chemical formula is extracted from the structure file using ASE
- If the name/formula doesn't match any entry in the fugacity table, the simulation falls back to ideal gas approximation

### Python Interface

You can also use the package programmatically, see example ./examples/ZIF8_CO2/run_gcmc_example.py

## Output Files

Simulations generate output files in the specified output directory (default: `results/`):

- **GCMC Isotherm (using `run_gcmc()`)**:
  - `isotherm_data.json`: Complete isotherm data (pressures, uptakes, energies, etc.)
  - `log_{pressure}bar.bin`: Binary log file containing all iteration data (step, uptake, interaction_energy, total_energy)
  - `restart/restart_{pressure}bar.xyz` and `.json`: Restart information (updated every step for crash recovery)
  - `checkpoints_{pressure}bar/checkpoint_{step}/`: History checkpoints saved at intervals specified by `--save-interval`
    - `traj.xyz`: Snapshot trajectory
    - `results.json`: Snapshot results (n_iter, uptake, interaction_energy, total_energy)
  
- **GCMC (direct class usage)**: 
  - `log_{pressure}bar.bin`: Binary log file with all iteration data
  - `restart/restart_{pressure}bar.xyz` and `.json`: Restart information
  - `checkpoints_{pressure}bar/checkpoint_{step}/`: History checkpoints
  
- **Widom**:
  - `widom_results.json`: Adsorption energies and calculated properties
  - `widom_traj.xyz`: Trajectory of valid insertions

### Isotherm Data Format

The `isotherm_data.json` file contains:
```json
{
    "temperature": 298.0,
    "pressures": [0.1, 0.5, 1.0, ...],
    "uptakes": [0.5, 1.2, 2.1, ...],
    "uptake_stds": [0.1, 0.2, 0.3, ...],
    "adsorption_energies": [-0.15, -0.18, -0.20, ...],
    "unit_cell_volume_A3": 1234.5,
    "unit_cell_volume_cm3": 1.234e-21,
    "adsorbent_file": "tests/zif8.xyz",
    "n_equilibration_steps": 10000,
    "n_production_steps": 20000
}
```

## Supported Compounds

The Peng-Robinson EOS supports the following compounds (via `PREOS.from_name()`):
- NH3 (ammonia)
- Ar (argon)
- C6H6 (benzene)
- CO2 (carbon dioxide)
- CH4 (methane)
- N2 (nitrogen)
- H2O (water)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mlip_mc,
  title = {MLIP-MC: Monte Carlo Simulations with Machine-Learned Interatomic Potentials},
  author = {Edwards, Connor W. and Yang, Fengxu and Stracke, Konstantin and Evans, Jack D.},
  year = {2025},
  license = {MIT}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Much of the code in this repository is based on or derived from the work published at:
- **Zenodo**: [10.5281/zenodo.7904959](https://doi.org/10.5281/zenodo.7904959)

Additional acknowledgments:
- Built on the ASE framework
- Supports neuqip models
