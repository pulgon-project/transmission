import pytest_datadir
from ase.io.vasp import read_vasp, write_vasp
import numpy as np
from ipdb import set_trace
from pulgon_tools_wip.utils import (
    fast_orth,
    get_character,
    get_matrices,
    get_matrices_withPhase,
    find_axis_center_of_nanotube,
    dimino_affine_matrix_and_subsquent,
    Cn,
    S2n,
    sigmaH,
    U,
    brute_force_generate_group_subsquent,
    get_symbols_from_ops
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol

from pymatgen.core.operations import SymmOp
import logging
from utilities import divide_irreps, divide_over_irreps, get_adapted_matrix


def test_family_6(shared_datadir):
    atom_center = read_vasp(shared_datadir / "F6")
    cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol

    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
    trans_op = cyclic.get_generators()
    rots_op = obj.get_generators()
    symbols = get_symbols_from_ops(rots_op)
    mats = [trans_op] + rots_op
    ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-4)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)
    matrices = get_matrices(atom_center, ops_car_sym)

    k_w = 5
    tmp1 = atom_center.positions[:, 2].reshape(-1, 1)
    factor_pos = tmp1 - tmp1.T
    factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
    matrices = get_matrices_withPhase(atom_center, ops_car_sym, k_w)
    # matrices = matrices * np.exp(1j * k_w * factor_pos)

    num_atoms = len(atom_center.numbers)
    DictParams = {"qpoints": k_w,"nrot": nrot, "order": order_ops, "family": family, "a": aL}
    basis, dims = get_adapted_matrix(DictParams, num_atoms, matrices)
    assert basis.shape == (3*len(atom_center), 3*len(atom_center)) and sum(dims) == 3*len(atom_center)


def test_family_8(shared_datadir):
    atom = read_vasp(shared_datadir / "F8")
    atom_center = find_axis_center_of_nanotube(atom)

    cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol

    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)

    trans_op = np.round(cyclic.get_generators(), 6)
    rots_op = obj.get_generators()
    mats = [trans_op] + rots_op
    symbols = get_symbols_from_ops(rots_op)

    ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-4)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)

    k_w = 5
    tmp1 = atom_center.positions[:, 2].reshape(-1, 1)
    factor_pos = tmp1 - tmp1.T
    factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
    matrices = get_matrices_withPhase(atom_center, ops_car_sym, k_w)
    matrices = matrices * np.exp(1j * k_w * factor_pos)

    num_atoms = len(atom_center.numbers)
    DictParams = {"qpoints": k_w,"nrot": nrot, "order": order_ops, "family": family, "a": aL}
    basis, dims = get_adapted_matrix(DictParams, num_atoms, matrices)
    assert basis.shape == (3*len(atom_center),3*len(atom_center)) and sum(dims) == 3*len(atom_center)

