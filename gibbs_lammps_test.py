from pathlib import Path
from ase import Atoms
from ase.data import vdw_radii
from mlip_mc.main import _load_model
from mlip_mc.src.gibbs import MLP_Gibbs

calc = _load_model(
    model_path=str(Path("examples/gibbs_co2_trappe/co2_trappe.json")),
    device="cpu",
    backend="lammps-classical",
)

# Rigid CO2 template in O-C-O order (matches intra_bonds in the JSON).
co2 = Atoms("OCO", positions=[(0, 0, 1.149), (0, 0, 0), (0, 0, -1.149)])

# Apply per-element charges so MLP_Gibbs's box assembly inherits them.
q = calc._mlipmc_charges
co2.set_initial_charges([q[s] for s in co2.get_chemical_symbols()])

sim = MLP_Gibbs(
    model=calc,
    atoms_mol=co2,
    T=240.0,
    N1_init=256, N2_init=256,
    L1_init=30.0, L2_init=30.0,
    device="cpu",
    vdw_radii=vdw_radii,
    n_equilibration_steps=1000000,
    n_production_steps=2000000,
    output_dir="results_gibbs_co2_trappe",
)
sim.run(3000000)
