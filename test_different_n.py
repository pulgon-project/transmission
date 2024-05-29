#!/usr/bin/env python
# Copyright 2018 Jesús Carrete Montaña <jesus.carrete.montana@tuwien.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os.path
import copy
import argparse

import tqdm
import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.linalg as la
from ase.io import read
from ase.io.vasp import write_vasp
import ase.data
import matplotlib
import matplotlib.pyplot as plt
import phonopy

import decimation
from ipdb import set_trace
from pulgon_tools_wip.utils import (
    fast_orth,
    get_character,
    get_matrices,
    find_axis_center_of_nanotube,
    dimino_affine_matrix_and_subsquent,
    Cn,
    S2n,
    sigmaH,
    brute_force_generate_group_subsquent,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz

from pymatgen.core.operations import SymmOp
import logging
from ase import Atoms
from utilities import divide_irreps, divide_over_irreps, get_adapted_matrix_multiq, get_adapted_matrix


path_atom = "datas/WS2/6-6-u1-3-defect-1/POSCAR"
# path_atom = "datas/WS2/6-6-u1-3-defect-1/POSCAR-1x1x3"

poscar_ase = read(path_atom)
cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
atom = cyclic._primitive

# atom_center = find_axis_center_of_nanotube(poscar_ase)
atom_center = poscar_ase

# write_vasp("test_poscar.vasp", atom_center)
aL = atom_center.cell[2,2]

################ family 4 ##################
family = 4
obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
nrot = obj.get_rotational_symmetry_number()
num_irreps = nrot * 2
sym = []
tran = SymmOp.from_rotation_and_translation(Cn(2 * nrot), [0, 0, 1 / 2])
# pg1 = obj.get_generators()
# sym.append(pg1[0])
rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
mirror = SymmOp.reflection([0, 0, 1], [0, 0, 0])
sym.append(tran.affine_matrix)
sym.append(rots.affine_matrix)
sym.append(mirror.affine_matrix)

################### family 2 #############
# family = 2
# num_irreps = 6
# obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
# nrot = obj.get_rotational_symmetry_number()
# sym = []
# # pg1 = obj.get_generators()  # change the order to satisfy the character table
# # sym.append(pg1[1])
# rots = SymmOp.from_rotation_and_translation(S2n(nrot), [0, 0, 0])
# sym.append(rots.affine_matrix)

#########################################
ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-4)

ops_car_sym = []
for op in ops:
    tmp_sym = SymmOp.from_rotation_and_translation(
        op[:3, :3], op[:3, 3] * aL
    )
    ops_car_sym.append(tmp_sym)

matrices = get_matrices(atom_center, ops_car_sym)
num_atoms = len(atom_center.numbers)

k_test = np.linspace(0, (np.pi - 0.1) / aL, 10, endpoint=True)

DictParams = {"nrot":nrot, "order":order_ops, "family":family, "a":aL}
adapteds_test, dimensions_test = get_adapted_matrix_multiq(k_test, DictParams, num_atoms, matrices)

set_trace()
