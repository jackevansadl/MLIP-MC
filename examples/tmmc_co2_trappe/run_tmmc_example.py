"""
TMMC example: bulk TraPPE CO2 vapor with the classical LAMMPS backend.

A single Transition-Matrix Monte Carlo run at one reference fugacity
yields the macrostate distribution ln Pi(N), which is reweighted to a
whole pressure grid — the complete <N>(P) curve from one simulation.
Because the system is a simple vapor, ln Pi is unimodal and the
reweighted isotherm is smooth (no hysteresis); the point of this example
is a fast, classical-forcefield validation of the TMMC machinery. See
the README for the flexible-framework (MLIP) recipe where hysteresis
actually appears.

Reuses the TraPPE CO2 setup from examples/gibbs_co2_trappe. MD moves
stay disabled: the classical setup holds molecules together with
zero-coefficient topology bonds (no intramolecular forces), so rigid
molecules are sampled with MC translation/rotation moves instead.

Run from the repository root:
    python examples/tmmc_co2_trappe/run_tmmc_example.py
"""
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import vdw_radii
from ase.units import bar

from mlip_mc.src.tmmc import MLP_TMMC
from mlip_mc.src.lammps_backend import build_lammps_calculator
from mlip_mc.src import tmmc_analysis

HERE = Path(__file__).parent
GIBBS_EXAMPLE = HERE.parent / "gibbs_co2_trappe"

# --- Simulation settings ------------------------------------------------
T = 300.0                 # K
P_REF_BAR = 20.0          # reference pressure the walker samples at
L = 30.0                  # cubic box edge (A)
N_MIN, N_MAX = 0, 60      # macrostate window (molecules)
N_STEPS = 200_000
BIAS_UPDATE = 5_000
OUTPUT_DIR = "results_tmmc_co2_trappe"

# --- Classical LAMMPS calculator (TraPPE CO2) ---------------------------
with open(GIBBS_EXAMPLE / "co2_trappe.json") as fh:
    config = json.load(fh)

calc = build_lammps_calculator(
    lmpcmds=GIBBS_EXAMPLE / config["lammps_input"],
    atom_types={k: int(v) for k, v in config["atom_types"].items()},
    keep_alive=bool(config.get("keep_alive", True)),
    create_box_extra=config.get("create_box_extra"),
    intra_bonds=config.get("intra_bonds"),
    lammps_threads=config.get("lammps_threads"),
)

# --- System: empty box + rigid linear CO2 template ----------------------
atoms_frame = Atoms(cell=[L, L, L], pbc=True)
atoms_ads = Atoms('OCO', positions=[[0.0, 0.0, -1.16],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.16]])

P_ref = P_REF_BAR * bar
fugacity = tmmc_analysis.fugacity_from_pressure(T, P_ref, molecule='CO2')
print(f"Reference pressure {P_REF_BAR} bar -> fugacity {fugacity/bar:.3f} bar")

tmmc = MLP_TMMC(
    model=calc,
    atoms_frame=atoms_frame,
    atoms_ads=atoms_ads,
    T=T,
    P=P_ref,
    fugacity=fugacity,
    device='cpu',
    vdw_radii=vdw_radii,
    N_min=N_MIN,
    N_max=N_MAX,
    bias_update_interval=BIAS_UPDATE,
    output_dir=OUTPUT_DIR,
    n_steps=N_STEPS,
)
tmmc.run(N=N_STEPS)

# --- Reweight to a full isotherm ----------------------------------------
N_grid = np.arange(N_MIN, N_MAX + 1)
pressures = list(np.logspace(np.log10(P_REF_BAR / 20), np.log10(P_REF_BAR * 2), 30))
isotherm = tmmc_analysis.compute_isotherm(
    lnPi_ref=tmmc.lnPi,
    N_grid=N_grid,
    T=T,
    pressures_bar=pressures,
    f_ref=fugacity,
    molecule='CO2',
)

out = Path(OUTPUT_DIR) / "isotherm_tmmc.json"
with open(out, 'w') as fh:
    json.dump(isotherm, fh, indent=4)
print(f"\nReweighted isotherm written to {out}")
print(f"{'P (bar)':>10} {'<N>':>8}")
for p, n in zip(isotherm['pressure_bar'], isotherm['mean_N']):
    print(f"{p:>10.3f} {n:>8.2f}")
