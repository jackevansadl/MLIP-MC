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
from widom import MLP_Widom

device = 'cuda' if torch.cuda.is_available() else 'cpu'

predictor = load_predict_unit("tests/uma-s-1p1.pt", device=device)
model = FAIRChemCalculator(predictor, task_name="odac")


atoms_frame = read('tests/zif8.xyz')
# C and O were renamed to Co and Os to differentiate them from framework atoms during training
atoms_ads = molecule('CO2')
cell = atoms_frame.cell
T = 273 


widom = MLP_Widom(model, atoms_frame, atoms_ads, T, device, vdw_radii)
widom.run(int(1000))

