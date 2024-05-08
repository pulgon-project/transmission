import pytest_datadir
from ase.io.vasp import read_vasp, write_vasp
import numpy as np
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
    U,
    brute_force_generate_group_subsquent,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer

from pymatgen.core.operations import SymmOp
import logging
from utilities import divide_irreps, divide_over_irreps, get_adapted_matrix_multiq, get_adapted_matrix


def test_family_1(shared_datadir):
    poscar_ase = read_vasp(shared_datadir / "F5")
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom = cyclic._primitive

    # atom_center = find_axis_center_of_nanotube(poscar_ase)
    atom_center = poscar_ase
    write_vasp("poscar.vasp", atom_center)

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    ################ family 4 ##################
    family = 5
    sym = []
    tran = SymmOp.from_rotation_and_translation(Cn(12).T, [0, 0, 1 / 3])

    rots1 = SymmOp.from_rotation_and_translation(Cn(4), [0, 0, 0])
    rots2 = SymmOp.from_rotation_and_translation(U(), [0, 0, 0])
    sym.append(tran.affine_matrix)
    sym.append(rots1.affine_matrix)
    sym.append(rots2.affine_matrix)

    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-6)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)

    set_trace()
    matrices = get_matrices(atom_center, ops_car_sym)
    num_atoms = len(atom_center.numbers)

    k_test = np.linspace(0, (np.pi - 0.1) / aL, 10, endpoint=True)
    adapteds_test, dimensions_test = get_adapted_matrix_multiq(k_test, nrot, order_ops, family, aL, num_atoms, matrices)



