import logging
import os.path
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import pretty_errors
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.utils import (
    get_matrices,
    get_matrices_withPhase,
    find_axis_center_of_nanotube,
    Cn,
    S2n,
    brute_force_generate_group_subsquent,
    get_symbols_from_ops
)
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol

from pymatgen.core.operations import SymmOp
from tqdm import tqdm
from utilities import get_adapted_matrix
import decimation
import ase
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "data_directory", help="directory")
    parser.add_argument(
        "-k",
        "--kpoints",
        type=int,
        default=4,
        help="plot the raw phonon or not",
    )

    args = parser.parse_args()
    path_0 = args.data_directory
    num_k = args.kpoints

    path_yaml = os.path.join(path_0, "phonopy.yaml")
    path_fc_continum = os.path.join(path_0, "force_constants.hdf5")
    path_save_band_yaml = os.path.join(path_0, "band_sym-adapted.yaml")
    path_anime = os.path.join(path_0, "test_anime")

    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)


    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)

    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    aL = poscar_ase.cell[2,2]
    atom = cyclic._atom
    atom_center = find_axis_center_of_nanotube(atom)

    NQS = num_k
    # k_start = -np.pi
    k_start = 0
    k_end = np.pi

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints_ori, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    qpoints = qpoints_ori[0]
    qpoints_1dim = qpoints[:,2] * 2 * np.pi
    qpoints_1dim = qpoints_1dim / aL


    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol

    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
    trans_op = cyclic.get_generators()
    rots_op = obj.get_generators()
    mats = [trans_op] + rots_op
    symbols = get_symbols_from_ops(rots_op)

    # ################ family 8 ###################
    family = 8
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    sym  = []
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    mirror = SymmOp.reflection([1,0,0], [0,0,0])
    sym.append(tran.affine_matrix)
    sym.append(rots.affine_matrix)
    sym.append(mirror.affine_matrix)


    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-4)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)
    num_atom = len(poscar_ase.numbers)
    phonon.run_band_structure(qpoints_ori, path_connections=connections, with_eigenvectors=True)

    frequencies, bands, eigvecs_convert = [], [], []
    num_atom = len(poscar_ase.numbers)
    for ii, qp in enumerate(tqdm(qpoints_1dim)):   # loop q points
        DictParams = {"qpoints":qp,  "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13
        # tmp1 = phonon.primitive.positions[:, 2].reshape(-1, 1)
        # factor_pos = tmp1 - tmp1.T
        # factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
        # if inv_index == True:
        #     factor_pos = factor_pos.T
        matrices = get_matrices_withPhase(atom_center, ops_car_sym, qp)
        # matrices = get_matrices_withPhase(atom_center, ops_car_sym, 0)
        # matrices = matrices * np.exp(1j * qp * factor_pos)
        adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)

        qz = qpoints[ii]
        D = phonon.get_dynamical_matrix_at_q(qz)
        D = adapted.conj().T @ D @ adapted
        start = 0
        tmp_band, tmp_eigvec = [], []
        for ir in range(len(dimensions)):
            end = start + dimensions[ir]
            block = D[start:end, start:end]
            eig, eigvecs = np.linalg.eigh(block)
            e = (
                    np.sqrt(np.abs(eig))
                    * np.sign(eig)
                    * VaspToTHz
            ).tolist()

            tmp_vec = adapted[:, start:end] @ eigvecs
            tmp_eigvec.append(tmp_vec)
            tmp_band.append(e)
            start = end
        bands.append(np.concatenate(tmp_band))
        eigvecs_convert.append(np.concatenate(tmp_eigvec, axis=1))
    bands = np.array(bands)# .swapaxes(0, 1) # * 2 * np.pi
    eigvecs_convert = np.array(eigvecs_convert)#.swapaxes(0, 1) # * 2 * np.pi

    phonon.band_structure._eigenvalues = bands
    phonon.band_structure._eigenvectors = eigvecs_convert
    # phonon.band_structure._frequencies = bands
    # phonon.band_structure.write_yaml(filename=path_save_band_yaml)

    for ii in range(len(bands[0])):
        path_band_ii = os.path.join(path_anime, "md_%d" % ii)
        if not os.path.exists(path_band_ii):
            os.makedirs(path_band_ii)
        path_band_file = os.path.join(path_band_ii, "APOSCAR")

        qpoints_index = 0
        q_point = qpoints[qpoints_index]
        phonon.write_animation(
            q_point,
            anime_type="poscar",
            band_index=ii+1,
            amplitude=80,
            num_div=50,
            shift=None,
            filename=path_band_file,
            qpoints_index=qpoints_index,
            eigenvalues=bands,
            eigenvectors=eigvecs_convert,
        )


if __name__ == "__main__":
    main()
