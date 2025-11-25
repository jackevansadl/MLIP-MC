import sys
import os
import numpy as np
import torch
torch.set_num_threads(6)
from ase.io import read
from ase.data import vdw_radii


from ase.units import bar
from ase.build import molecule
np.random.seed(42)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

# hacky way to import things
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)
from gcmc import MLP_GCMC

PRESSURE = 1.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

predictor = load_predict_unit("tests/uma-s-1p1.pt", device=device)
model = FAIRChemCalculator(predictor, task_name="odac")


atoms_frame = read('tests/zif8.xyz')
# C and O were renamed to Co and Os to differentiate them from framework atoms during training
atoms_ads = molecule('CO2')
cell = atoms_frame.cell
T = 273 


for pressure in [PRESSURE]:
    print(f"--- Initializing GCMC simulation for {pressure} bar ---")

    P = pressure * bar
    from utilities import PREOS
    eos = PREOS.from_name('carbondioxide')
    fugacity = eos.calculate_fugacity(T,P) # outputs units in same units as inputted pressure ?
    # print(f"Fugacity at {pressure} bar and {T} K: {fugacity / bar} bar")
    gcmc = MLP_GCMC(model, atoms_frame, atoms_ads, T, P, fugacity, device, vdw_radii, debug=True)

    print(f"--- Starting GCMC simulation for {pressure} bar ---")
    gcmc.run(int(100))
    print(f"--- Finished GCMC simulation for {pressure} bar ---")
