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
import time
from turtledemo.forest import start

import tqdm
import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.linalg as la
import ase.io
import ase.data
import matplotlib
import matplotlib.pyplot as plt
import phonopy
from ase.io.vasp import write_vasp
from scipy.linalg import eigvals
from sympy import factor

import decimation
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
    brute_force_generate_group_subsquent,
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz

from pymatgen.core.operations import SymmOp
import logging
from ase import Atoms
from utilities import counting_y_from_xy, get_adapted_matrix, divide_irreps, divide_over_irreps, get_modified_adapted_matrix
import matplotlib.colors as mcolors


matplotlib.rcParams["font.size"] = 16.0
NPOINTS = 21

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the transmission across a defective Si nanowire"
    )
    parser.add_argument(
        "-e",
        "--eps",
        type=float,
        default=1e-8,
        help="prefactor for the imaginary part of the energies",
    )
    parser.add_argument(
        "-t1",
        "--rtol",
        type=float,
        default=1e-4,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
             " it a propagating mode",
    )
    parser.add_argument(
        "-t2",
        "--atol",
        type=float,
        default=1e-3,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
             " it a propagating mode",
    )
    parser.add_argument(
        "-t3",
        "--mtol",
        type=float,
        default=1e-2,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
             " it a propagating mode",
    )
    parser.add_argument(
        "-t4",
        "--means_tol",
        type=float,
        default=1e-2,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
             " it a propagating mode",
    )
    parser.add_argument(
        "-d",
        "--decimation",
        type=float,
        default=1e-10,
        help="tolerance for the decimation procedure",
    )
    parser.add_argument(
        "-m",
        "--maxiter",
        type=int,
        default=100000,
        help="maximum number of iterations in the decimation loop",
    )
    # parser.add_argument("phonopy_file", help="phonopy yaml file")
    # parser.add_argument("pure_fc_file", help="force constant file")
    # parser.add_argument("scatter_fc_file", help="force constant file")
    # parser.add_argument("defect_indices", help="force constant file")
    parser.add_argument("data_directory", help="directory")
    args = parser.parse_args()

    print("*******************")
    rcond = 0.2
    print("rcond =", rcond)
    print("*******************")

    t0 = time.time()

    path_directory = args.data_directory
    path_phonopy_defect = os.path.join(path_directory, "phonopy_defect.yaml")
    path_phonopy_pure = os.path.join(path_directory, "phonopy_pure.yaml")
    path_LR_blocks = os.path.join(path_directory, "pure_fc.npz")
    path_fc_continum = os.path.join(path_directory, "FORCE_CONSTANTS_pure.continuum")

    path_scatter_blocks = os.path.join(path_directory, "scatter_fc.npz")
    path_defect_indices = os.path.join(path_directory, "defect_indices.npz")
    path_poscar = os.path.join(path_directory, "POSCAR")
    path_savedata = os.path.join(path_directory, "transmission_irreps.npz")

    path_save_phonon = os.path.join(path_directory, "phonon_dec_dyn")

    ######################### projector ##########################
    # phonon = phonopy.load(phonopy_yaml=path_phonopy_pure, is_compact_fc=True)
    # phonon = phonopy.load(phonopy_yaml=path_phonopy_pure, force_constants_filename=path_fc_continum, is_compact_fc=True)
    phonon = phonopy.load(phonopy_yaml=path_phonopy_pure, force_constants_filename=path_fc_continum)
    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    # atom = cyclic._primitive
    atom = poscar_ase
    atom_center = find_axis_center_of_nanotube(atom)

    ################ family 8 ######################
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

    ################## family 6 ########################
    # family = 6
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.rot_sym[0][1]
    # num_irreps = int(nrot/2)+1
    # sym  = []
    # rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    # sym.append(rots.affine_matrix)
    # sym.append(obj.get_generators()[1])
    # ##############################################

    ops, order_ops = brute_force_generate_group_subsquent(sym)
    if len(ops) != len(order_ops):
        logging.ERROR("len(ops) != len(order)")

    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * cyclic._pure_trans
        )
        ops_car_sym.append(tmp_sym)

    # matrices = get_matrices(atom_center, ops_car_sym)


    num_atoms = len(phonon.primitive.numbers)
    LR_blocks = np.load(path_LR_blocks)
    scatter_blocks = np.load(path_scatter_blocks)
    defect_indices = np.load(path_defect_indices)["defect_indices"]
    poscar = ase.io.read(path_poscar)
    phonon_defect = phonopy.load(phonopy_yaml=path_phonopy_defect, is_compact_fc=True)

    HL = LR_blocks["H00"]
    TL = LR_blocks["H01"]
    KC = scatter_blocks["Hc"]
    VLC = scatter_blocks["Vlc"]
    VCR = scatter_blocks["Vcr"]

    cells = max(defect_indices) + 1
    idx_scatter = np.where(defect_indices == int((cells - 1) / 2))[0]

    mass_C = np.diag(np.power(ase.data.atomic_masses[phonon_defect.primitive.numbers[idx_scatter]], -1 / 2))
    mass_L = np.diag(np.power(ase.data.atomic_masses[poscar.numbers], -1 / 2))
    mass_R = mass_L.copy()
    unit = 1e-24 * ase.units.m ** 2 * ase.units.kg / ase.units.J
    # unit = 1

    HL1 = np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, HL), mass_L)
    TL1 = np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, TL), mass_L)

    HL = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, HL), mass_L)
    TL = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, TL), mass_L)
    KC = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_C, KC), mass_C)
    VLC = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, VLC), mass_C)
    VCR = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_C, VCR), mass_R)

    HL1 = HL1.transpose((0, 2, 1, 3)).reshape((HL1.shape[0] * 3, -1))
    TL1 = TL1.transpose((0, 2, 1, 3)).reshape((TL1.shape[0] * 3, -1))

    HL = HL.transpose((0, 2, 1, 3)).reshape((HL.shape[0] * 3, -1))
    TL = TL.transpose((0, 2, 1, 3)).reshape((TL.shape[0] * 3, -1))
    KC = KC.transpose((0, 2, 1, 3)).reshape((KC.shape[0] * 3, -1))
    VLC = VLC.transpose((0, 2, 1, 3)).reshape((VLC.shape[0] * 3, -1))
    VCR = VCR.transpose((0, 2, 1, 3)).reshape((VCR.shape[0] * 3, -1))

    HR = HL.copy()
    TR = TL.copy()
    aL = phonon.primitive.cell[2, 2]
    aR = aL

    # HL_complex = HL.astype(complex)
    # TL_complex = TL.astype(complex)
    # HR_complex = HR.astype(complex)
    # TR_complex = TR.astype(complex)
    #
    # iomega, omega = 0, 15.846153846153847
    # inv_gLretm = decimation.inv_g00(
    #     HL_complex, TL_complex, omega, args.eps, args.decimation, args.maxiter
    # )
    # eigvals0, eigvecs0 = np.linalg.eigh(inv_gLretm)
    # k_w_group = np.angle(eigvals0) / aL

    path = [[[0, 0, -0.5], [0, 0, 0.5]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NPOINTS
    )
    qpoints = qpoints[0]
    qpoints_1dim = qpoints[:,2] * 2 * np.pi
    qpoints_1dim = qpoints_1dim / aL


    # Plot the phonon spectra of both bulk leads.
    qvec = np.linspace(-np.pi, np.pi, num=NPOINTS, endpoint=True)
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    # Compute all the parameters of the interface
    inc_omega = np.linspace(1e-1, omegaL.max() * 1.01, num=NPOINTS, endpoint=True)

    frequencies1, frequencies2, frequencies3 = [], [], []
    for ii, qp in enumerate(qpoints_1dim):   # loop q points
        qz = qpoints[ii]
        D1 = phonon.get_dynamical_matrix_at_q(qz)
        eigvals1, eigvecs1 = np.linalg.eigh(D1)
        eigvals1 = eigvals1.real
        frequencies1.append(np.sqrt(abs(eigvals1)) * np.sign(eigvals1) * VaspToTHz)

        # D2 = TL1.conj().transpose() * np.exp(-1j*qp*aL) + HL1 + TL1 * np.exp(1j*qp*aL)


        D2 = TL1.conj().transpose() * np.exp(-1j*qp*aL) + HL1 + TL1 * np.exp(1j*qp*aL)
        # D2 = D2 * np.exp(1j*qp*factor_pos)
        # set_trace()

        eigvals2, eigvecs2 = np.linalg.eigh(D2)
        eigvals2 = eigvals2.real
        frequencies2.append(np.sqrt(abs(eigvals2)) * np.sign(eigvals2) * VaspToTHz)

        res = np.round(np.angle(np.exp(1j * (np.angle(D1)-np.angle(D2)))), 5)
        DictParams = {"nrot": nrot, "order": order_ops, "family": family, "a": aL, "qpoints": qp}  # F:2,4, 13

        tmp1 = phonon.primitive.positions[:,2].reshape(-1,1)
        factor_pos = tmp1 - tmp1.T
        factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1)
        matrices = get_matrices_withPhase(atom_center, ops_car_sym, qp)
        matrices = matrices * np.exp(1j*qp*factor_pos)

        # set_trace()

        basis, dims = get_adapted_matrix(DictParams, num_atoms, matrices)
        # basis, dims = get_modified_adapted_matrix(DictParams, num_atoms, matrices)

        D_sym = basis.conj().T @ D2 @ basis
        start = 0
        adapted_vecs, adapteds_values, adapteds_freq = [], [], []
        for dim in dims:
            end = start + dim
            eigvals3, eigvecs3 = np.linalg.eigh(D_sym[start:end, start:end])

            adapteds_values.append(eigvals3)
            adapted_vecs.append(basis[:, start:end] @ eigvecs3)
            eigvals3 = eigvals3.real
            adapteds_freq.append(np.sqrt(abs(eigvals3)) * np.sign(eigvals3) * VaspToTHz * 2 * np.pi)
            start = np.copy(end)
        frequencies3.append(np.concatenate(adapteds_freq))
        # adapteds_values = np.array(adapteds_values)
        # adapted_vecs = np.array(adapted_vecs, axis=1)

    frequencies1 = np.array(frequencies1).T * 2 * np.pi
    frequencies2 = np.array(frequencies2).T * 2 * np.pi
    frequencies3 = np.array(frequencies3).swapaxes(0, 1)

    fig, ax = plt.subplots()
    for i, f_raw in enumerate(frequencies1):
        if i == 0:
            ax.plot(qpoints_1dim, f_raw, color="grey", label="raw1")
            ax.plot(qpoints_1dim, frequencies2[i], color="blue", label="raw2")
            # ax.plot(qpoints_1dim, omegaL.T[i], color="red", label="decimation")
        else:
            ax.plot(qpoints_1dim, f_raw, color="grey")
            ax.plot(qpoints_1dim, frequencies2[i], color="blue")
            # ax.plot(qpoints_1dim, omegaL.T[i], color="red")

    #### plot adapted phonon
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'yellow', 'pink', 'olive', 'sage', 'slategray', 'darkkhaki', 'yellowgreen']
    # color = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    labels = ["|m|=0","|m|=1","|m|=2","|m|=3","|m|=4","|m|=5","|m|=6","|m|=7","|m|=8","|m|=9", "|m|=10", "|m|=11","|m|=12", "|m|=13", "|m|=14"]

    dim_sum = np.cumsum(dims)
    for ii, freq in enumerate(frequencies3):
        idx_ir = (ii > dim_sum - 1).sum()
        if ii in dim_sum-1:
            ax.plot(qpoints_1dim, freq, label=labels[int(abs(idx_ir))], color=color[int(abs(idx_ir))])
        else:
            ax.plot(qpoints_1dim, freq, color=color[int(abs(idx_ir))])

    plt.xlabel("qpoints_onedim")
    plt.ylabel("frequencies * 2 * pi (Thz)")
    plt.tight_layout()
    # plt.legend()
    # plt.savefig(path_save_phonon, dpi=600)
    plt.show()

