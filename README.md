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

### Command-Line Interface

#### GCMC Isotherm Simulation

```bash
# Multiple pressure points (auto-distributed across GPUs)
mlip_mc \\
    --mode gcmc \\
    --adsorbent framework.xyz \\
    --adsorbate-molecule CO2 \\
    --temperature 298.0 \\
    --pressures 0.1,0.5,1.0,2.0,5.0,10.0,20.0 \\
    --n-equil 10000 \\
    --n-prod 20000 \\
    --output-dir results
```

#### Widom Insertion

```bash
# Basic Widom insertion calculation
mlip_mc \\
    --mode widom \\
    --adsorbent framework.xyz \\
    --adsorbate-molecule CO2 \\
    --temperature 298.0 \\
    --n-trials 10000 \\
    --output-dir widom_results
```

**Command-Line Arguments:**
- `--mode`: Simulation mode: `gcmc` (Grand Canonical Monte Carlo) or `widom` (Widom insertion) (default: `gcmc`)
- `--adsorbent`: Path to adsorbent structure file (.xyz, .cif, etc.) **(required)**
- `--adsorbate-path`: Path to adsorbate structure file (optional). The chemical formula will be automatically extracted to match with the fugacity table.
- `--adsorbate-molecule`: Molecule name (e.g., CO2, CH4) if not using file. This name will be used to match with the fugacity table.
- `--temperature`: Temperature in Kelvin **(required)**
- `--pressures`: Comma-separated pressures in bar, or single number **(required for GCMC mode)**
- `--n-equil`: Number of equilibration steps for GCMC (default: 10000)
- `--n-prod`: Number of production steps for GCMC (default: 20000)
- `--n-trials`: Number of Widom insertion trials (default: 10000)
- `--save-interval`: Interval for saving history checkpoints in GCMC (default: 1000)
- `--model`: Path to MLIP model file. Can be a local path (default: `models/model.pt`) or a Hugging Face repository name like `fengxuyoung/MLIP-MC` (or `hf://fengxuyoung/MLIP-MC`). Missing files are automatically downloaded and cached. The model format should match your installed backend (FAIRChem `.pt` files or MACE `.model` files).
- `--output-dir`: Output directory (default: results)
- `--hf-token`: Hugging Face access token for downloading private models or bypassing interactive login (optional)
- `--gpu-id`: GPU device ID to use (for Widom mode, default: 0, use -1 for CPU)

**Model caching:** Hugging Face downloads are cached under `~/.cache/mlip-mc/<repo>/<filename>` (or a custom directory set via the `MLIP_MC_CACHE` environment variable). Subsequent runs reuse the cached file even when launched from different working directories.

**Note:** The adsorbate name for EOS (fugacity) calculation is automatically determined:
- If `--adsorbate-molecule` is provided, that name is used to match with the fugacity table
- If `--adsorbate-path` is provided, the chemical formula is extracted from the structure file using ASE
- If the name/formula doesn't match any entry in the fugacity table, the simulation falls back to ideal gas approximation

### Python Interface

You can also use the package programmatically for more control and integration into your workflows:

#### GCMC Isotherm

```python
from mlip_mc import run_gcmc

# Run GCMC simulation
results = run_gcmc(
    adsorbent_path="framework.xyz",
    adsorbate_molecule="CO2",
    temperature=298.0,
    pressure_points=[0.1, 1.0, 5.0],
    n_equilibration_steps=10000,
    n_production_steps=20000,
    model_path="fengxuyoung/MLIP-MC",
    output_dir="results"
)

# Access results
print(f"Pressures: {results['pressures']}")
print(f"Uptakes: {results['uptakes']}")
print(f"Temperature: {results['temperature']} K")
```

#### Widom Insertion

```python
from mlip_mc import run_widom

# Run Widom insertion calculation
results = run_widom(
    adsorbent_path="framework.xyz",
    adsorbate_molecule="CO2",
    temperature=298.0,
    n_trials=10000,
    model_path="",
    output_dir="widom_results"
)
```

## Output Files

Simulations generate output files in the specified output directory (default: `results/`):

- **GCMC Isotherm (using `run_gcmc()`)**:
  - `isotherm_data.json`: Complete isotherm data (pressures, uptakes, energies, etc.)
  - `log_{pressure}bar.bin`: Binary log file containing all iteration data with trajectory (step, uptake, interaction_energy, total_energy, atomic structure)
  - `restart/restart_{pressure}bar.xyz` and `.json`: Restart information (updated every step for crash recovery)
  - `checkpoints_{pressure}bar/checkpoint_{step}/`: History checkpoints saved at intervals specified by `--save-interval`
    - `traj.xyz`: Snapshot trajectory
    - `results.json`: Snapshot results (n_iter, uptake, interaction_energy, total_energy)
  
- **GCMC (direct class usage)**: 
  - `log_{pressure}bar.bin`: Binary log file with all iteration data
  - `restart/restart_{pressure}bar.xyz` and `.json`: Restart information
  - `checkpoints_{pressure}bar/checkpoint_{step}/`: History checkpoints
  
- **Widom Insertion**:
  - `widom_results.json`: Adsorption energies and calculated properties (Henry's constant, weighted average energy, etc.)
  - `log_widom.bin`: Binary log file containing all valid insertions with trajectory data (trial number, adsorption energy, total energy, atomic structure) - one record per valid insertion
  - `widom_trajectory.xyz`: Last valid insertion structure saved at the end of simulation
  - `restart/restart_widom.xyz` and `.json`: Restart information (updated every step for crash recovery)

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
