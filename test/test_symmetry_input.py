import unittest
from pulgon_tools_wip.utils import get_symbols_from_ops, brute_force_generate_group_subsquent, get_matrices
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from ase.io.vasp import read_vasp
from pymatgen.core.operations import SymmOp
import pytest_datadir
from ipdb import set_trace


def test_family_num(shared_datadir):
    atom_center = read_vasp(shared_datadir / "F6")
    cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol
    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
    assert family==6


def test_generator_symbols(shared_datadir):
    atom_center = read_vasp(shared_datadir / "F6")
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    mats = obj.get_generators()
    symbols = get_symbols_from_ops(mats)
    assert symbols[0] == 'C5' and symbols[1] == 'sigmaV'


def test_generate_full_group(shared_datadir):
    atom_center = read_vasp(shared_datadir / "F6")
    cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    trans_op = cyclic.get_generators()
    rots_op = obj.get_generators()
    mats = [trans_op] + rots_op
    ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-4)
    assert len(ops) == len(order_ops) == 10


def test_get_matrices(shared_datadir):
    atom_center = read_vasp(shared_datadir / "F6")
    aL = atom_center.cell[2, 2]
    cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    trans_op = cyclic.get_generators()
    rots_op = obj.get_generators()
    mats = [trans_op] + rots_op
    ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-4)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)
    matrices = get_matrices(atom_center, ops_car_sym)
    assert len(matrices) == 10


if __name__ == '__main__':
    unittest.main()
