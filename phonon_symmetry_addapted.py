import copy
import logging
import os.path
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import pretty_errors
import scipy.linalg as la
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.utils import (
    fast_orth,
    get_character,
    get_matrices,
    find_axis_center_of_nanotube,
    dimino_affine_matrix_and_subsquent,
    Cn,
    sigmaH,
    sigmaV,
    S2n,
    U,
    atom_move_z,
    brute_force_generate_group_subsquent,
    brute_force_generate_group,
    affine_matrix_op,
)
from pymatgen.core.operations import SymmOp
from tqdm import tqdm
from utilities import counting_y_from_xy, get_adapted_matrix


def main():
    path_0 = "datas/WS2/6-6-u1-3-defect-1"
    path_yaml = os.path.join(path_0, "phonopy_pure.yaml")
    path_fc_continum = os.path.join(path_0, "FORCE_CONSTANTS_pure.continuum")
    path_save_phonon = os.path.join(path_0, "phonon_defect_sym_adapted")
    path_save_phonon_transmission = os.path.join(path_0, "transmission_pure_from_phonon")

    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)
    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)

    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom = cyclic._primitive
    atom_center = find_axis_center_of_nanotube(atom)

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()

    NQS = 51
    # k_start = -np.pi+0.1
    k_start = 0
    k_end = np.pi-0.1

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    qpoints = qpoints[0]

    qpoints_1dim = np.linspace(k_start/2/np.pi, k_end/2/np.pi, num=NQS, endpoint=k_end)
    # qpoints_1dim = np.linspace(k_start, k_end, num=NQS, endpoint=k_end)
    qpoints_1dim = qpoints_1dim / cyclic._pure_trans

    sym = []
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    sym.append(tran.affine_matrix)
    pg1 = obj.get_generators()    # change the order to satisfy the character table
    # sym.append(pg1[1])
    rot = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    sym.append(rot.affine_matrix)
    mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    sym.append(mirror.affine_matrix)

    # sym.append(affine_matrix_op(pg1[0], pg1[1]))
    # sym.append(SymmOp.from_rotation_and_translation(Cn(6), [0,0,0]).affine_matrix)
    # sym.append(SymmOp.reflection(normal=[0,0,1], origin=atom_center.get_scaled_positions()[2]).affine_matrix)
    ops, order = brute_force_generate_group_subsquent(sym, symec=1e-6)

    if len(ops) != len(order):
        logging.ERROR("len(ops) != len(order)")
    # ops = np.round(ops, 8)

    ops_car_sym = []
    for op in ops:
        tmp_sym1 = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * cyclic._pure_trans
        )
        ops_car_sym.append(tmp_sym1)
    matrices = get_matrices(atom_center, ops_car_sym)
    # set_trace()
    # matrices = get_matrices(atom, ops_car_sym)

    family = 4
    # characters, paras_values, paras_symbols = get_character(qpoints_1dim, nrot, order, family, a=cyclic._pure_trans)
    # characters = np.array(characters)
    # characters = characters[::2] + characters[1::2]
    # paras_values = np.array(paras_values)[::2]
    # paras_values = np.array(paras_values)


    frequencies, distances, bands = [], [], []
    num_atom = len(atom.numbers)
    for ii, qp in enumerate(tqdm(qpoints_1dim)):   # loop q points

        adapted, dimensions = get_adapted_matrix(qp, nrot, order, family, cyclic._pure_trans, num_atom, matrices)
        qz = qpoints[ii]

        D = phonon.get_dynamical_matrix_at_q(qz)
        D = adapted.conj().T @ D @ adapted
        start = 0
        tmp_band = []
        for ir in range(len(dimensions)):
            end = start + dimensions[ir]
            block = D[start:end, start:end]
            eig = la.eigvalsh(block)
            e = (
                    np.sqrt(np.abs(eig))
                    * np.sign(eig)
                    * VaspToTHz
                    # * phonopy.units.VaspToEv
                    # * 1e3
            ).tolist()
            tmp_band.append(e)

            start = end
        bands.append(tmp_band)

        if ii == 0:
            distances.append(0)
            q_last = qz.copy()
        else:
            distances.append(
                np.linalg.norm(
                    np.dot(qz - q_last, phonon.supercell.get_cell())
                )
            )
    frequencies = (
        np.array(bands).swapaxes(0, 1).swapaxes(1, 2)
    )

    # %%
    frequencies_raw = []
    for ii, q in enumerate(qpoints):
        D = phonon.get_dynamical_matrix_at_q(q)
        eigvals, eigvecs = np.linalg.eigh(D)
        eigvals = eigvals.real
        frequencies_raw.append(np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz)
    frequencies_raw = np.array(frequencies_raw).T

    fig, ax = plt.subplots()
    #### raw phonon
    for i, f_raw in enumerate(frequencies_raw):
        if i == 0:
            ax.plot(distances, f_raw, color="grey", label="raw")
        else:
            ax.plot(distances, f_raw, color="grey")

    #### symmetry adapted
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'yellow', 'pink']
    labels = ["|m|=0","|m|=1","|m|=2","|m|=3","|m|=4","|m|=5","|m|=6","|m|=7","|m|=8","|m|=9"]

    if family==2:
        for ii, freq in enumerate(frequencies):
            for jj, fq in enumerate(freq):
                if jj==0 and (ii>=int(nrot/2) - 1):
                    ax.plot(np.array(distances), fq, label=labels[int(abs(ii-nrot/2+1))], color=color[int(abs(ii-nrot/2+1))])
                else:
                    ax.plot(np.array(distances), fq, color=color[int(abs(ii-nrot/2+1))])
    elif family==4:
        for ii, freq in enumerate(frequencies):
            for jj, fq in enumerate(freq):
                if jj==0 and (ii>=int(nrot) - 1):
                    ax.plot(np.array(distances), fq, label=labels[int(abs(ii-nrot+1))], color=color[int(abs(ii-nrot+1))])
                else:
                    ax.plot(np.array(distances), fq, color=color[int(abs(ii-nrot+1))])

    # plt.xlim(0, 0.6)  # x轴刻度范围
    # plt.ylim(-2, 2)  # x轴刻度范围
    plt.xlabel("distances")
    plt.ylabel("frequencies Thz")
    plt.legend()
    # plt.savefig(path_save_phonon, dpi=600)


    fig1, ax1 = plt.subplots()
    frequencies = frequencies * 2 * np.pi

    # pairs = []
    # for im, freq in enumerate(frequencies):
    #     pair = []
    #     for jj, fq in enumerate(freq):
    #         pair.append([fq.min(), fq.max()])
    #     pairs.append(pair)
    # pairs = np.array(pairs)
    # pairs_all = np.concatenate(pairs, axis=0)


    x = np.linspace(0, frequencies.max(), num=NQS)
    y = np.zeros((len(x)))
    for ii, omega in enumerate(x):
        counts = 0
        for jj, freq in enumerate(frequencies):
            counts += counting_y_from_xy(omega, freq)
        y[ii] = counts
    ax1.plot(x, y, label="sum_all", color="grey")


    ym = []
    for im, freq in enumerate(frequencies):
        y = np.zeros((len(x)))
        for ii, omega in enumerate(x):
            counts = counting_y_from_xy(omega, freq)
            y[ii] = counts
        ym.append(y)


    # ym = []
    # for im, pair in enumerate(pairs):
    #     y = np.zeros((len(x)))
    #     for ii, omega in enumerate(x):
    #         counts = 0
    #         for jj, pa in enumerate(pair):
    #             min_val, max_val = pa
    #             if omega>min_val and omega<max_val:
    #                 counts += 1
    #         y[ii] = counts
    #     ym.append(y)

    if family==2:
        ym_abs = np.array([ym[2],ym[1]+ym[3],ym[0]+ym[4],ym[5]])    #  family 2
        for im, y in enumerate(ym_abs):
            ax1.plot(x, y, label=labels[im], color=color[im])
    elif family==4:
        ym_abs = np.array([ym[5],ym[4]+ym[6],ym[3]+ym[7],ym[2]+ym[8],ym[1]+ym[9],ym[0]+ym[10], ym[11]])    #  family 4
        for im, y in enumerate(ym_abs):
            ax1.plot(x, y, label=labels[im], color=color[im])


    plt.xlabel("Frequencies * 2pi")
    plt.ylabel("Counts")
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)

    plt.legend()
    # plt.savefig(path_save_phonon_transmission, dpi=600)

    plt.show()

if __name__ == "__main__":
    main()
