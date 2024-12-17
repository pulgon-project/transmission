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
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer

from pymatgen.core.operations import SymmOp
import logging
from utilities import divide_irreps, divide_over_irreps, get_adapted_matrix


def test_family_4(shared_datadir):
    poscar_ase = read_vasp(shared_datadir / "F4")
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom_center = poscar_ase
    # atom_center = find_axis_center_of_nanotube(poscar_ase)

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]

    ################# family 4 #########################
    family = 4
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    num_irreps = nrot * 2
    sym = []
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    sym.append(tran.affine_matrix)
    rot = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    sym.append(rot.affine_matrix)
    mirror = SymmOp.reflection([0,0,1], [0,0,0.5])
    sym.append(mirror.affine_matrix)
    ####################################################
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-6)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)

    k_w = 0
    tmp1 = atom_center.positions[:, 2].reshape(-1, 1)
    factor_pos = tmp1 - tmp1.T
    factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
    matrices = get_matrices_withPhase(atom_center, ops_car_sym, k_w)
    matrices = matrices * np.exp(1j * k_w * factor_pos)

    num_atoms = len(atom_center.numbers)
    DictParams = {"qpoints": k_w,"nrot": nrot, "order": order_ops, "family": family, "a": aL}
    basis, dims = get_adapted_matrix(DictParams, num_atoms, matrices)
    assert basis.shape == (3*len(atom_center),3*len(atom_center)) and sum(dims) == 3*len(atom_center)


def test_family_6(shared_datadir):
    poscar_ase = read_vasp(shared_datadir / "F6")
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom = cyclic._primitive

    # atom_center = find_axis_center_of_nanotube(poscar_ase)
    atom_center = poscar_ase
    write_vasp("poscar.vasp", atom_center)

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]

    ################## family 6 ########################
    family = 6
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.rot_sym[0][1]
    num_irreps = int(nrot/2)+1
    sym  = []
    rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    sym.append(rots.affine_matrix)
    sym.append(obj.get_generators()[1])
    ####################################################
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-6)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)

    matrices = get_matrices(atom_center, ops_car_sym)

    k_w = 1
    tmp1 = atom_center.positions[:, 2].reshape(-1, 1)
    factor_pos = tmp1 - tmp1.T
    factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
    matrices = get_matrices_withPhase(atom_center, ops_car_sym, k_w)
    matrices = matrices * np.exp(1j * k_w * factor_pos)

    num_atoms = len(atom_center.numbers)
    DictParams = {"qpoints": k_w,"nrot": nrot, "order": order_ops, "family": family, "a": aL}
    basis, dims = get_adapted_matrix(DictParams, num_atoms, matrices)
    assert basis.shape == (3*len(atom_center),3*len(atom_center)) and sum(dims) == 3*len(atom_center)


def test_family_8(shared_datadir):
    poscar_ase = read_vasp(shared_datadir / "F8")
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom = cyclic._primitive

    # atom_center = find_axis_center_of_nanotube(poscar_ase)
    atom_center = poscar_ase
    write_vasp("poscar.vasp", atom_center)

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]

    ################## family 6 ########################
    family = 8
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    sym  = []
    num_irreps = nrot + 1
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    # tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/4])
    rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    mirror = SymmOp.reflection([1,0,0], [0,0,0])
    sym.append(tran.affine_matrix)
    sym.append(rots.affine_matrix)
    sym.append(mirror.affine_matrix)
    ####################################################
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-6)
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

