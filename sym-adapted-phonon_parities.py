import logging
import os.path
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import pretty_errors
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from matplotlib.pyplot import legend
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
    sigmaH,
    brute_force_generate_group_subsquent,
    get_symbols_from_ops
)
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol
from pymatgen.core.operations import SymmOp
from sparse import astype
from tqdm import tqdm
from utilities import  get_adapted_matrix, get_adapted_matrix_withparities
import decimation
from spglib import get_symmetry_dataset
import ase
import argparse
import matplotlib.colors as mcolors
from pymatgen.core.operations import SymmOp
import seaborn as sns


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
    path_save_phonon = os.path.join(path_0, "phonon_defect_sym_adapted_paraties")
    path_savedata = os.path.join(path_0, "sym-adapted-phonon_parities")
    path_poscar = os.path.join(path_0, "POSCAR")

    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)
    # phonon = phonopy.load(phonopy_yaml=path_yaml, is_compact_fc=True)
    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)
    # poscar_ase = read_vasp(path_poscar)

    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    aL = poscar_ase.cell[2,2]
    atom = cyclic._atom
    atom_center = find_axis_center_of_nanotube(atom)

    NQS = num_k
    k_start = -np.pi + 1e-3
    # k_start = np.pi
    # k_start = 0
    # k_start = 0.1
    k_end = np.pi - 1e-3

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
    if family==13:
        nrot = np.int32(nrot / 2)

    trans_op = np.round(cyclic.get_generators(), 6)
    rots_op = np.round(obj.get_generators(), 6)
    mats = np.vstack(([trans_op], rots_op))
    # mats = np.vstack(([trans_op], [rots_op]))
    # symbols = get_symbols_from_ops([rots_op])
    symbols = get_symbols_from_ops(rots_op)

    # set_trace()

    ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-2)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)

    distances, bands, dimensions_tol, paras_values_tol, paras_symbols_tol = [], [], [], [], []
    num_atom = len(poscar_ase.numbers)
    for ii, qp in enumerate(tqdm(qpoints_1dim)):   # loop q points
        DictParams = {"qpoints":qp,  "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13
        # DictParams = {"qpoints":0,  "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13

        # tmp1 = phonon.primitive.positions[:, 2].reshape(-1, 1)
        # factor_pos = tmp1 - tmp1.T
        # factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
        # if inv_index == True:
        #     factor_pos = factor_pos.T
        # matrices = get_matrices(atom_center, ops_car_sym, symprec=1e-5)
        matrices = get_matrices_withPhase(atom_center, ops_car_sym, qp, symprec=1e-2)
        # matrices = get_matrices_withPhase(atom_center, ops_car_sym, 0, symprec=1e-2)
        # matrices = matrices * np.exp(1j * qp * factor_pos)
        # adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)
        adapted, dimensions, paras_values, paras_symbols = get_adapted_matrix_withparities(DictParams, num_atom, matrices)
        dimensions_tol.append(dimensions)
        paras_values_tol.append(paras_values)
        paras_symbols_tol.append(paras_symbols)

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
    color = plt.cm.tab10.colors
    # color = plt.cm.tab20.colors

    # palette1 = sns.color_palette("tab20") + sns.color_palette("Set3")
    # color = palette1[:30]

    labels = []
    for ii in range(40):
        labels.append("|m|=%d" %ii)

    if family==8 or family==6:
        dim_sum = np.cumsum(dimensions)
        for ii, freq in enumerate(frequencies):
            idx_ir = (ii > dim_sum - 1).sum()
            if ii in dim_sum-1:
                ax.plot(np.array(distances), freq, label=r"$m=%d, \Pi_{v}=%d$" % (int(paras_values[idx_ir][1]), int(paras_values[idx_ir][2])), color=color[int(abs(idx_ir))])
            else:
                ax.plot(np.array(distances), freq, color=color[int(abs(idx_ir))])

    elif family==13:
        for ii, freq in enumerate(frequencies.T):
            dim_sum = np.cumsum(dimensions_tol[ii])
            tmp_value = paras_values_tol[ii]
            tmp_symbol = paras_symbols_tol[ii]

            for jj in range(len(freq)):
                idx_ir = (jj > dim_sum - 1).sum()
                if ii == 0 and dim_sum[idx_ir] - 1 == jj:
                    ax.scatter(distances[ii], freq[jj], label=r"$m=%d, \Pi_{U}=%d, \Pi_{V}=%d, \Pi_{H}=%d, $" % (
                    int(tmp_value[idx_ir][1]), int(tmp_value[idx_ir][2]), int(tmp_value[idx_ir][3]), int(tmp_value[idx_ir][4])), color=color[int(abs(idx_ir))])
                elif qpoints_1dim[ii] == 0 and dim_sum[idx_ir] - 1 == jj:
                    ax.scatter(distances[ii], freq[jj], label=r"$m=%d, \Pi_{U}=%d, \Pi_{V}=%d, \Pi_{H}=%d, $" % (
                    int(tmp_value[idx_ir][1]), int(tmp_value[idx_ir][2]), int(tmp_value[idx_ir][3]), int(tmp_value[idx_ir][4])), color=color[int(abs(idx_ir)+nrot+3)])
                else:
                    ax.scatter(distances[ii], freq[jj], color=color[int(abs(idx_ir))])

    np.savez(path_savedata, distances=np.array(distances), frequencies=frequencies, dim_sum=dim_sum)
    labelsize = 14
    fontsize = 16
    legendsize = 8

    plt.xlabel("q", fontsize=fontsize)
    plt.ylabel(r"$\omega$" + " (Thz)", fontsize=fontsize)
    # plt.legend(fontsize=10, loc="best")
    plt.legend(fontsize=legendsize, loc="upper right")

    plt.tight_layout()
    plt.tick_params(labelsize=labelsize)

    plt.savefig(path_save_phonon, dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()
