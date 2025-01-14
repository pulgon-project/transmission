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

from utilities import  get_adapted_matrix
import decimation
from spglib import get_symmetry_dataset
import ase
import argparse
import matplotlib.colors as mcolors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "data_directory", help="directory")
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        default=False,
        help="plot the raw phonon or not",
    )
    parser.add_argument(
        "-k",
        "--kpoints",
        type=int,
        default=11,
        help="plot the raw phonon or not",
    )

    args = parser.parse_args()
    path_0 = args.data_directory
    raw = args.raw
    num_k = args.kpoints

    path_yaml = os.path.join(path_0, "phonopy_pure.yaml")
    path_fc_continum = os.path.join(path_0, "FORCE_CONSTANTS_pure.continuum")
    path_save_phonon = os.path.join(path_0, "phonon_defect_sym_adapted")
    path_savedata = os.path.join(path_0, "sym-adapted-phonon")

    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)
    # phonon = phonopy.load(phonopy_yaml=path_yaml, is_compact_fc=True)
    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)

    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    aL = poscar_ase.cell[2,2]
    atom = cyclic._atom
    atom_center = find_axis_center_of_nanotube(atom)

    NQS = num_k
    k_start = -np.pi
    # k_start = 0
    k_end = np.pi

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    qpoints = qpoints[0]
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



    # ############### family 4 ##################
    # family = 4
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.get_rotational_symmetry_number()
    # sym  = []
    # tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    # rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    # mirror = SymmOp.reflection([0,0,1], [0,0,0.75])
    # sym.append(tran.affine_matrix)
    # sym.append(rots.affine_matrix)
    # sym.append(mirror.affine_matrix)
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
    # ################## family 2 ###################
    # family = 2
    # num_irreps = 6
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.get_rotational_symmetry_number()
    # sym = []
    # # pg1 = obj.get_generators()  # change the order to satisfy the character table
    # # sym.append(pg1[1])
    # rots = SymmOp.from_rotation_and_translation(S2n(nrot), [0, 0, 0])
    # sym.append(rots.affine_matrix)
    # ############### family 6 ######################
    # family = 6
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.rot_sym[0][1]
    # sym  = []
    # rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    # # mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    # # sym.append(tran.affine_matrix)
    # sym.append(rots.affine_matrix)
    # sym.append(obj.get_generators()[1])
    # ################################################
    # ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-4)
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-4)

    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)
    # matrices = get_matrices(atom_center, ops_car_sym)



    frequencies, distances, bands = [], [], []
    num_atom = len(poscar_ase.numbers)
    for ii, qp in enumerate(tqdm(qpoints_1dim)):   # loop q points
        DictParams = {"qpoints":qp,  "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13

        tmp1 = phonon.primitive.positions[:, 2].reshape(-1, 1)
        factor_pos = tmp1 - tmp1.T
        factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
        # if inv_index == True:
        #     factor_pos = factor_pos.T
        matrices = get_matrices_withPhase(atom_center, ops_car_sym, qp)
        # matrices = get_matrices_withPhase(atom_center, ops_car_sym, 0)
        # matrices = matrices * np.exp(1j * qp * factor_pos)
        adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)

        qz = qpoints[ii]
        D = phonon.get_dynamical_matrix_at_q(qz)
        # D = TL.conj().transpose() * np.exp(-1j*qp*aL) + HL + TL * np.exp(1j*qp*aL)
        D = adapted.conj().T @ D @ adapted

        start = 0
        tmp_band = []
        for ir in range(len(dimensions)):
            end = start + dimensions[ir]
            block = D[start:end, start:end]
            eig, eigvecs = np.linalg.eigh(block)
            e = (
                    np.sqrt(np.abs(eig))
                    * np.sign(eig)
                    * VaspToTHz
            ).tolist()
            tmp_band.append(e)
            start = end
        bands.append(np.concatenate(tmp_band))
        distances.append(qp * aL)
    frequencies = np.array(bands).swapaxes(0, 1) # * 2 * np.pi

    fig, ax = plt.subplots()
    if raw == True:
        frequencies_raw = []
        for ii, qp in enumerate(tqdm(qpoints_1dim)):   # loop q points
            qz = qpoints[ii]
            D = phonon.get_dynamical_matrix_at_q(qz)
            # D = TL.conj().transpose() * np.exp(-1j*qp*aL) + HL + TL * np.exp(1j*qp*aL)

            eigvals, eigvecs = np.linalg.eigh(D)
            eigvals = eigvals.real
            frequencies_raw.append(np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz)
        frequencies_raw = np.array(frequencies_raw).T  # * 2 * np.pi
        ### raw phonon
        for i, f_raw in enumerate(frequencies_raw):
            if i == 0:
                ax.plot(distances, f_raw, color="grey", label="raw")
            else:
                ax.plot(distances, f_raw, color="grey")

    #### plot adapted phonon
    # color = [value for key, value in mcolors.XKCD_COLORS.items()]
    # color = plt.cm.tab10.colors
    color = plt.cm.tab20.colors
    labels = []
    for ii in range(40):
        labels.append("|m|=%d" %ii)
        dim_sum = np.cumsum(dimensions)

    if family==4:
        for ii, freq in enumerate(frequencies):
            idx_ir = (ii > dim_sum - 1).sum()
            if ii in dim_sum-1 and idx_ir>=nrot - 1:
                ax.plot(np.array(distances), freq, label=labels[int(abs(idx_ir-nrot+1))], color=color[int(abs(idx_ir-nrot+1))])
            else:
                ax.plot(np.array(distances), freq, color=color[int(abs(idx_ir-nrot+1))])
    elif family==2:
        for ii, freq in enumerate(frequencies):
            idx_ir = (ii > dim_sum - 1).sum()
            if ii in dim_sum-1 and idx_ir>=nrot/2 - 1:
                ax.plot(np.array(distances), freq, label=labels[int(abs(idx_ir-nrot/2+1))], color=color[int(abs(idx_ir-nrot/2+1))])
            else:
                ax.plot(np.array(distances), freq, color=color[int(abs(idx_ir-nrot/2+1))])
    elif family==6:
        for ii, freq in enumerate(frequencies):
            idx_ir = (ii > dim_sum - 1).sum()
            if ii in dim_sum-1:
                ax.plot(np.array(distances), freq, label=labels[int(abs(idx_ir))], color=color[int(abs(idx_ir))])
            else:
                ax.plot(np.array(distances), freq, color=color[int(abs(idx_ir))])
    elif family==8:
        for ii, freq in enumerate(frequencies):
            idx_ir = (ii > dim_sum - 1).sum()
            if ii in dim_sum-1:
                ax.plot(np.array(distances), freq, label=labels[int(abs(idx_ir))], color=color[int(abs(idx_ir))])
            else:
                ax.plot(np.array(distances), freq, color=color[int(abs(idx_ir))])
    np.savez(path_savedata, distances=np.array(distances), frequencies=frequencies, dim_sum=dim_sum)

    # plt.xlabel("$k\cdot a$" + " ($\AA$)")
    labelsize = 14
    fontsize = 16

    plt.xlabel("q", fontsize=fontsize)
    plt.ylabel(r"$\omega$" + " (Thz)", fontsize=fontsize)
    # plt.legend(fontsize=10, loc="best")
    plt.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.tick_params(labelsize=labelsize)

    plt.savefig(path_save_phonon, dpi=500)
    plt.show()


if __name__ == "__main__":
    main()
