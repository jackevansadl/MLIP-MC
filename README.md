# MLIP-MC

ASE framework for Monte Carlo simulations with universal Machine-Learned Interatomic Potentials (MLIP).

## Overview

MLIP-MC is a Python package for performing Monte Carlo simulations of gas adsorption in porous materials using machine-learned interatomic potentials. The package integrates seamlessly with the ASE (Atomic Simulation Environment) framework and supports MLIP models from both **FAIRChem** and **MACE-Torch** backends.

## Installation

### Backend Selection

MLIP-MC supports two MLIP backends. You must install one of them:

- **FAIRChem**: For models trained with FAIRChem (e.g., OC20, OC22 models)
- **MACE-Torch**: For MACE models (e.g., MACE-MP models)

### Quick Install

Install MLIP-MC with your preferred backend:

**With FAIRChem backend:**
```bash
# rocm install: pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
pip install ".[fairchem]"
```

**With MACE-Torch backend:**
```bash
# rocm install: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install ".[mace-torch]"
```

**With Orbital backend:**
```bash
# rocm install: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip install ".["orb-models"]"
```

**Development mode (includes test tooling):**
```bash
pip install -e ".[BACKEND_OF_YOUR_CHOICE,dev]"
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
- `--model`: Path to MLIP model file. Can be a local path (default: `models/model.pt`) or a Hugging Face repository name like `fengxuyoung/MLIP-MC` (or `hf://fengxuyoung/MLIP-MC`). Missing files are automatically downloaded and cached. The model format should match your installed backend (FAIRChem `.pt` files or MACE `.model` files).
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

`H2 (hydrogen)`    `He (helium)`        `NH3 (ammonia)`      `H2O (water)`  
`CH4 (methane)`    `N2 (nitrogen)`      `O2 (oxygen)`        `Ar (argon)`  
`CO (carbon monoxide)` `CO2 (carbon dioxide)` `C2H2 (acetylene)`  `C2H6 (ethane)`  
`C3H8 (propane)`   `C4H10 (butane)`     `C6H6 (benzene)`     `C6H14 (n-hexane)`

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
- Supports MLIP models from FAIRChem and MACE-Torch
