# Gibbs ensemble CO2 with TraPPE (LAMMPS classical backend)

Reproduces the RASPA3 example [`examples/advanced/1_mc_gibbs_co2`](https://github.com/iRASPA/RASPA3/tree/main/examples/advanced/1_mc_gibbs_co2):
two 30 A cubic boxes, 256 rigid CO2 each, T = 240 K, TraPPE-style LJ + Coulomb
with PPPM Ewald, 12 A real-space cutoff.

## Files

- `co2_trappe.lmp` - raw LAMMPS commands (pair_style, bond_style, special_bonds, kspace, charges).
- `co2_trappe.json` - backend config consumed by `_load_model(..., backend='lammps-classical')`,
  including `create_box_extra` (extra args to `create_box` for bond support) and `intra_bonds`
  (which atom-index pairs inside each rigid molecule are bonded).

## Requirements

- `lammps` Python module (bindings) with `KSPACE` package enabled.
- `ase` (for `LAMMPSlib`).

See `Dockerfile.lammps` at the repo root for a working build.

## Bonded force-field setup (how it works)

ASE's bundled `LAMMPSlib` doesn't declare bonds or molecule IDs, so `lj/cut/coul/long`
otherwise computes the (huge) 1-2 / 1-3 LJ + Coulomb terms inside each rigid CO2.
We avoid that with three coordinated pieces:

1. **`create_box_extra`** (in `co2_trappe.json`) — injects
   `bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 4` into the auto
   `create_box` command issued by LAMMPSlib, so the data structures for bonds
   are allocated.
2. **`intra_bonds`** (in `co2_trappe.json`) — describes the rigid molecule's
   bond topology: 3 atoms per molecule, bonds `(0,1)` (O-C) and `(1,2)` (C-O).
   The wrapper issues `create_bonds single/bond` commands for every molecule
   after each LAMMPSlib `(re)build`, so MC moves that grow or shrink the
   system keep bonds consistent.
3. **`special_bonds lj 0 0 0 coul 0 0 0`** (in `co2_trappe.lmp`) — zeros out
   1-2 (bonded) and 1-3 (next-nearest, via the central C) pair terms, so the
   intra-molecular LJ + Coulomb at r << sigma drops out of the pair sum.

Verified numerically:

| System | Energy (kcal/mol) |
|--------|------------------:|
| 1 rigid CO2 alone | +0.0023 (PPPM grid noise) |
| 2 rigid CO2, 21 A apart | +0.0036 (still ~zero) |
| 2 rigid CO2, 4 A apart, parallel | -0.078 (small inter-molecular attraction) |

## Atom ordering (important)

The bonded backend assumes ASE atoms are arranged in groups of 3 in
**O-C-O order** (atoms-per-molecule = 3, bonds at indices `(0,1)` and `(1,2)`).
The CO2 template in the snippet below produces this ordering naturally.
If you change the molecule template, update `intra_bonds.bond_pairs`
correspondingly.

## Running (Python API)

The CLI is not wired for this backend yet; use the Python API directly.

```python
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
    n_equilibration_steps=500,
    n_production_steps=1000,
    output_dir="results_gibbs_co2_trappe",
)
sim.run(1500)
```

Bump `n_equilibration_steps` / `n_production_steps` to match the RASPA3
reference (10k init / 5k equil / 10k prod) for a full comparison.
