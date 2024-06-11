from ase.io.vasp import read_vasp, write_vasp
import numpy as np
from ase import Atoms
from ipdb import set_trace

path_0 = "datas/molecular/CH4/H4C"
atom = read_vasp(path_0)

pos = (atom.positions - atom.positions[4]) @ np.linalg.inv(16*np.eye(3)) + [0.5,0.5,0.5]

new_atom = Atoms(numbers=atom.numbers, cell=16*np.eye(3), scaled_positions=pos)

save_name = "datas/molecular/CH4/CH4.vasp"
write_vasp(path_0, new_atom)
