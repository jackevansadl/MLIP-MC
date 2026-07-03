# Transition-Matrix Monte Carlo (TMMC) examples

TMMC replaces the one-pressure-per-simulation GCMC workflow with a
single flat-histogram simulation at one reference fugacity. It
accumulates the collection matrix `C(N -> N')` of unbiased transition
probabilities over the macrostate `N` (number of adsorbed molecules),
reconstructs the macrostate distribution `ln Pi(N)`, and uses it as a
bias so the whole window `[N_min, N_max]` is sampled uniformly — across
free-energy barriers that trap ordinary GCMC. Reweighting

```
ln Pi_f(N) = ln Pi_fref(N) + N ln(f / f_ref)
```

then produces the complete isotherm `<N>(P)` from that one run. When
the framework is flexible and `ln Pi` is bimodal (e.g. breathing MOFs,
capillary condensation of water), the two basins are the metastable
states seen on adsorption and desorption: TMMC predicts both hysteresis
branches and the equilibrium step pressure. This is the methodology of
Goeminne & Van Speybroeck, *J. Am. Chem. Soc.* 2025
(doi:10.1021/jacs.4c15287), built on the TMMC formulation of Errington,
*J. Chem. Phys.* 118, 9915 (2003).

## 1. Fast classical validation: bulk TraPPE CO2

`run_tmmc_example.py` runs bulk CO2 vapor with the classical LAMMPS
backend (setup shared with `examples/gibbs_co2_trappe`). It is cheap
enough to validate the TMMC machinery end-to-end: the reweighted
`<N>(P)` should follow the (near-ideal) vapor equation of state over
the whole pressure grid from a single run.

```bash
python examples/tmmc_co2_trappe/run_tmmc_example.py
```

Outputs in `results_tmmc_co2_trappe/`:

- `lnPi_*.json` — running `ln Pi(N)`, visit histogram, collection matrix
- `results_tmmc_*.json` — final distribution
- `isotherm_tmmc.json` — reweighted isotherm
- `restart/` — resumable state (rerunning the script continues the run)

Note: MD moves stay disabled with this backend. The classical setup
holds CO2 together with zero-coefficient topology bonds (no
intramolecular forces), so molecules are rigid and sampled with MC
translation/rotation.

## 2. Flexible framework with an MLIP (paper-style)

For a real adsorption problem — where flexibility and hysteresis
matter — use the `run_tmmc` entry point with an MLIP backend. Framework
flexibility is sampled with hybrid MC/MD moves: short MD trajectories
of the full system (framework + adsorbates) with resampled
Maxwell-Boltzmann velocities, interleaved between insertion/deletion
moves.

```bash
mlip_mc \
    --mode tmmc \
    --adsorbent framework.cif \
    --adsorbate-molecule H2O \
    --temperature 298.0 \
    --reference-pressure 0.02 \
    --n-steps 500000 \
    --n-min 0 --n-max 120 \
    --bias-update-interval 10000 \
    --md-probability 0.02 \
    --md-steps 1000 --md-timestep 0.5 \
    --model hf://your-org/your-mlip \
    --output-dir tmmc_h2o
```

or from Python:

```python
from mlip_mc import run_tmmc

results = run_tmmc(
    adsorbent_path='framework.cif',
    adsorbate_molecule='H2O',
    temperature=298.0,
    reference_pressure=0.02,       # bar; pick near the isotherm step
    n_steps=500_000,
    N_min=0, N_max=120,
    md_probability=0.02,           # hybrid MC/MD flexibility moves
    md_ensemble='nvt',             # 'npt' also samples cell fluctuations
    model_path='path/to/model',
    output_dir='tmmc_h2o',
)
print(results['isotherm']['transition_pressure_bar'])
```

`tmmc_results.json` contains `ln Pi(N)`, the reweighted equilibrium
isotherm, the metastable branch loadings `N_low` (adsorption branch)
and `N_high` (desorption branch) wherever `ln Pi` is bimodal, and the
equilibrium transition pressure where both basins carry equal weight.

Practical notes:

- Choose `N_max` comfortably above the saturation loading; the visit
  histogram (`H` in the results) shows whether the window was covered.
- `--md-ensemble npt` requires a backend with stress support and an
  upper-triangular cell; alternatively add MC volume moves with
  `--volume-probability 0.01` for global cell fluctuations.
- The run is restartable: the collection matrix, `ln Pi`, and the
  configuration are checkpointed in `restart/` and re-running with the
  same parameters continues toward the target step count.
