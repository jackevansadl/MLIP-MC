# MLIP-MC

ASE framework for Monte Carlo simulations with universal Machine-Learned Interatomic Potentials (MLIP).

## Overview

MLIP-MC is a Python package for performing Monte Carlo simulations of gas adsorption in porous materials using machine-learned interatomic potentials. The package integrates seamlessly with the ASE (Atomic Simulation Environment) framework and supports MLIP models from **FAIRChem**, **MACE-Torch** and **Orbital** backends.

## Installation

### Backend Selection

MLIP-MC supports three MLIP backends. You must install one of them:

- **FAIRChem**: For models trained with FAIRChem (e.g., OC20, OC22 models)
- **MACE-Torch**: For MACE models (e.g., MACE-MP models)
- **Orbital**: For Orbital models (e.g. orb_v3_conservative_inf_omat)

### Install from PyPI (Recommended)

Install MLIP-MC with your preferred backend directly from PyPI:

**With FAIRChem backend:**
```bash
pip install "mlip-mc[fairchem]"
```

**With MACE-Torch backend:**
```bash
pip install "mlip-mc[mace-torch]"
```

**With Orbital backend:**
```bash
pip install "mlip-mc[orb-models]"
```

> **ROCm users:** Install the ROCm version of PyTorch first before installing MLIP-MC:
> ```bash
> pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
> ```

### Install from Source

Clone the repository and install locally:

```bash
git clone https://github.com/jackevansadl/MLIP-MC.git
cd MLIP-MC
pip install ".[fairchem]"   # or ".[mace-torch]" or ".[orb-models]"
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
    --model models/model.pt \\
    --output-dir results \\
    --checkpoint-interval 10000 \\
    --write-trajectory \\
    --trajectory-interval 100
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
    --model models/model.pt \\
    --output-dir widom_results
```

#### Transition-Matrix Monte Carlo (TMMC)

TMMC computes the macrostate probability distribution ln Π(N) at a single
reference fugacity with a flat-histogram bias, then reweights it to every
pressure — the complete isotherm from one simulation. With a flexible
framework (hybrid MC/MD moves), a bimodal ln Π yields both metastable
adsorption/desorption branches (hysteresis) and the equilibrium step
pressure, following Goeminne & Van Speybroeck, *JACS* 2025
(doi:10.1021/jacs.4c15287).

```bash
# Full water isotherm with a flexible framework from a single run
mlip_mc \\
    --mode tmmc \\
    --adsorbent framework.cif \\
    --adsorbate-molecule H2O \\
    --temperature 298.0 \\
    --reference-pressure 0.02 \\
    --n-steps 500000 \\
    --n-min 0 --n-max 120 \\
    --md-probability 0.02 \\
    --model models/model.pt \\
    --output-dir tmmc_results
```

**Command-Line Arguments:**
- `--mode`: Simulation mode: `gcmc` (Grand Canonical Monte Carlo), `widom` (Widom insertion), or `tmmc` (Transition-Matrix Monte Carlo) (default: `gcmc`)
- `--adsorbent`: Path to adsorbent structure file (.xyz, .cif, etc.) **(required)**
- `--adsorbate-path`: Path to adsorbate structure file (optional). The chemical formula will be automatically extracted to match with the fugacity table.
- `--adsorbate-molecule`: Molecule name (e.g., CO2, CH4) if not using file. This name will be used to match with the fugacity table.
- `--temperature`: Temperature in Kelvin **(required)**
- `--pressures`: Comma-separated pressures in bar, or single number **(required for GCMC mode)**
- `--n-equil`: Number of equilibration steps for GCMC (default: 10000)
- `--n-prod`: Number of production steps for GCMC (default: 20000)
- `--n-trials`: Number of Widom insertion trials (default: 10000)
- `--checkpoint-interval`: Interval for saving history checkpoints in GCMC (default: 10000)
- `--model`: Path to MLIP model file **(required)**. Can be a local path or a Hugging Face URI (e.g., `hf://your-org/your-repo` or `hf://your-org/your-repo:model.pt`). When using the `hf://` scheme, files are automatically downloaded and cached. The model format should match your installed backend (FAIRChem `.pt` files or MACE `.model` files).
- `--output-dir`: Output directory (default: results)
- `--write-trajectory`: Specifies to write out simulation trajectory every `--trajectory-interval` steps (default: False)
- `--trajectory-interval`: Interval for saving structures into trajectory file (default: 100)
- `--hf-token`: Hugging Face access token for downloading private models or bypassing interactive login (optional)
- `--gpu-id`: GPU device ID to use (for Widom mode, default: 0, use -1 for CPU)
- `overwrite_checkpoints`: Specifies to overwrite checkpoint files every `--checkpoint-interval` steps (default: False)

**TMMC-specific arguments (`--mode tmmc`):**
- `--reference-pressure`: Pressure in bar at which ln Π is sampled (default: 1.0). Any value works; one near the isotherm step gives the best statistics.
- `--n-steps`: Total number of TMMC steps (default: 100000)
- `--n-min`, `--n-max`: Macrostate window — the range of adsorbed-molecule counts to sample (defaults: 0, 100). Choose `--n-max` above saturation loading.
- `--bias-update-interval`: Steps between ln Π bias refreshes from the collection matrix (default: 10000)
- `--md-probability`: Probability of hybrid MC/MD framework-flexibility moves (default: 0.0 = rigid framework)
- `--md-ensemble`: `nvt` (default) or `npt` for the hybrid MD moves (`npt` requires stress support and an upper-triangular cell)
- `--md-steps`, `--md-timestep`: MD trajectory length and timestep in fs per hybrid move (defaults: 1000, 1.0)
- `--volume-probability`: Probability of MC volume moves for cell fluctuations (default: 0.0)
- `--pressures`: Optional comma-separated pressures in bar for the reweighted isotherm (default: 40 points spanning `reference-pressure` / 100 to × 100)

**Model caching:** When using the `hf://` scheme, downloads are cached under `~/.cache/mlip-mc/<repo>/<filename>` (or a custom directory set via the `MLIP_MC_CACHE` environment variable). Subsequent runs reuse the cached file even when launched from different working directories.

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
    model_path="models/model.pt",  # or use hf://your-org/your-repo
    output_dir="results",
    write_trajectory=True,
    trajectory_interval=100,
    checkpoint_interval=10000
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
    model_path="models/model.pt",  # or use hf://your-org/your-repo
    output_dir="widom_results"
)
```

#### Transition-Matrix Monte Carlo

```python
from mlip_mc import run_tmmc

results = run_tmmc(
    adsorbent_path="framework.cif",
    adsorbate_molecule="H2O",
    temperature=298.0,
    reference_pressure=0.02,       # bar
    n_steps=500_000,
    N_min=0, N_max=120,
    md_probability=0.02,           # hybrid MC/MD flexibility moves
    model_path="models/model.pt",
    output_dir="tmmc_results",
)

isotherm = results['isotherm']
print(isotherm['pressure_bar'], isotherm['mean_N'])
# Hysteresis: metastable branches and equilibrium step pressure
print(isotherm['N_low'], isotherm['N_high'])
print(isotherm['transition_pressure_bar'])
```

See `examples/tmmc_co2_trappe/` for a fast classical-forcefield example
and the flexible-framework recipe.

## Output Files

Simulations generate output files in the specified output directory (default: `results/`):

- **GCMC Isotherm (using `run_gcmc()`)**:
  - `isotherm_data.json`: Complete isotherm data (pressures, uptakes, energies, etc.)
  - `log_{pressure}bar.bin`: Binary log file containing all iteration data with trajectory (step, uptake, interaction_energy, total_energy)
  - `traj_{pressure}bar.xyz`: Simulation trajectory
  - `restart/restart_{pressure}bar.xyz` and `.json`: Restart information (updated every step for crash recovery)
  - `checkpoints_{pressure}bar/checkpoint_{step}/`: History checkpoints saved at intervals specified by `--save-interval`
    - `traj.xyz`: Snapshot trajectory
    - `results.json`: Snapshot results (n_iter, uptake, interaction_energy, total_energy)
  
- **GCMC (direct class usage)**: 
  - `log_{pressure}bar.bin`: Binary log file with all iteration data
  - `restart/restart_{pressure}bar.xyz` and `.json`: Restart information
  - `checkpoints_{pressure}bar/checkpoint_{step}/`: History checkpoints
  - `traj_{pressure}bar.xyz`: simulation trajectory
  
- **Widom Insertion**:
  - `widom_results.json`: Adsorption energies and calculated properties (Henry's constant, weighted average energy, etc.)
  - `log_widom.bin`: Binary log file containing all valid insertions with trajectory data (trial number, adsorption energy, total energy, atomic structure) - one record per valid insertion
  - `widom_trajectory.xyz`: Last valid insertion structure saved at the end of simulation
  - `restart/restart_widom.xyz` and `.json`: Restart information (updated every step for crash recovery)

- **TMMC (using `run_tmmc()`)**:
  - `tmmc_results.json`: ln Π(N), visit histogram, reweighted isotherm with hysteresis branches (`N_low`/`N_high`), and equilibrium transition pressure
  - `lnPi_{pressure}bar.json`: Running ln Π estimate, visit histogram, and collection matrix (refreshed at every bias update)
  - `results_tmmc_{pressure}bar.json`: Final distribution and collection matrix
  - `log_tmmc_{pressure}bar.bin`: Binary log of accepted moves (same layout as the GCMC log; readable with `read_binary_log`)
  - `restart/restart_tmmc_{pressure}bar.xyz` and `.json`: Restart information including the collection matrix (a rerun with the same parameters resumes toward the target step count)

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
- Supports MLIP models from FAIRChem, MACE-Torch and Orbital
