import spglib
from ase.io.vasp import read_vasp
from ipdb import set_trace


atom = read_vasp("datas/WS2/6-6-u1-3-defect-1/POSCAR")

symmetry = spglib.get_symmetry(atom)

rotations = symmetry['rotations']
translations = symmetry['translations']

set_trace()
