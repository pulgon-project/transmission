from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol
import argparse
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
from ase.io.vasp import read_vasp
import pretty_errors
import logging


# logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the transmission across a defective Si nanowire"
    )

    parser.add_argument("data_poscar", help="poscar")
    parser.add_argument("tol", help="tolerance")


    args = parser.parse_args()

    path_poscar = args.data_poscar
    tol = float(args.tol)


    poscar_ase = read_vasp(path_poscar)
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)

    atom = cyclic._primitive
    atom_center = find_axis_center_of_nanotube(atom)

    obj = LineGroupAnalyzer(atom_center, tolerance=tol)
    nrot = obj.get_rotational_symmetry_number()

    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol
    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
    

    print("family=", family)
    print("generalized translation:", cyclic.cyclic_group)
    print("axial point group:", obj.sch_symbol)

