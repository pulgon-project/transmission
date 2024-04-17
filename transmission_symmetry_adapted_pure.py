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

import decimation
from ipdb import set_trace
from pulgon_tools_wip.utils import (
    fast_orth,
    get_character,
    get_matrices,
    find_axis_center_of_nanotube,
    dimino_affine_matrix_and_subsquent,
    Cn,
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
from utilities import counting_y_from_xy


def orthogonalize(values, vectors):
    modules = np.abs(values)
    phases = np.angle(values)
    order = np.argsort(-modules)
    values = np.copy(values[order])
    vectors = np.copy(vectors[:, order])
    lo = 0
    hi = 1
    groups = []
    while True:
        if hi >= vectors.shape[1] or not np.isclose(
                np.angle(values[hi]), np.angle(values[hi - 1]), args.tolerance
        ):
            groups.append((lo, hi))
            lo = hi
        if hi >= vectors.shape[1]:
            break
        hi += 1
    for g in groups:
        lo, hi = g
        if hi > lo + 1:
            values[lo:hi] = values[lo:hi].mean()
            vectors[:, lo:hi] = la.orth(vectors[:, lo:hi])
    return values, vectors



matplotlib.rcParams["font.size"] = 16.0
NPOINTS = 50


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the transmission across a defective Si nanowire"
    )
    parser.add_argument(
        "-e",
        "--eps",
        type=float,
        default=1e-5,
        help="prefactor for the imaginary part of the energies",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1e-3,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
        " it a propagating mode",
    )
    parser.add_argument(
        "-d",
        "--decimation",
        type=float,
        default=1e-8,
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

    path_directory = args.data_directory
    path_phonopy = os.path.join(path_directory, "phonopy_defect.yaml")
    path_phonopy_pure = os.path.join(path_directory, "phonopy_pure.yaml")
    path_LR_blocks = os.path.join(path_directory, "pure_fc.npz")
    path_scatter_blocks = os.path.join(path_directory, "scatter_fc.npz")
    path_defect_indices = os.path.join(path_directory, "defect_indices.npz")
    path_poscar = os.path.join(path_directory, "POSCAR")

    phonon = phonopy.load(phonopy_yaml=path_phonopy)
    poscar = ase.io.read(path_poscar)


    ######################### projector
    phonon = phonopy.load(phonopy_yaml=path_phonopy_pure, is_compact_fc=True)

    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom = cyclic._primitive
    atom_center = find_axis_center_of_nanotube(atom)


    k_start = -np.pi
    k_end = np.pi

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NPOINTS
    )
    qpoints = qpoints[0]

    qpoints_1dim = np.linspace(k_start, k_end, num=NPOINTS, endpoint=k_end)
    qpoints_1dim = qpoints_1dim / cyclic._pure_trans


    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()

    sym = []
    # tran = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 1])
    # sym.append(tran.affine_matrix)
    pg1 = obj.get_generators()    # change the order to satisfy the character table
    sym.append(pg1[1])
    ops, order = brute_force_generate_group_subsquent(sym)
    if len(ops) != len(order):
        logging.ERROR("len(ops) != len(order)")

    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * cyclic._pure_trans
        )
        ops_car_sym.append(tmp_sym)
    matrices = get_matrices(atom_center, ops_car_sym)

    family = 2
    characters, paras_values, paras_symbols = get_character(qpoints_1dim, nrot, order, family, a=cyclic._pure_trans)

    characters = np.array(characters)
    characters = characters[::2] + characters[1::2]
    paras_values = np.array(paras_values)[::2]

    frequencies, distances, bands = [], [], []
    ndof = 3 * len(atom.numbers)
    for ii, qp in enumerate(qpoints_1dim):   # loop q points
        idx = np.where(np.isclose(paras_values[:,0], qp))[0]
        if len(idx)!=int(len(characters)/NPOINTS):
            set_trace()
            logging.ERROR("the number of quantum incorrect")

        qz = qpoints[ii]
        D = phonon.get_dynamical_matrix_at_q(qz)
        dimensions, adapted = [], []
        remaining_dof = copy.deepcopy(ndof)

        for jj in idx:     # loop quantum number
            chara = characters[jj]
            projector = np.zeros((ndof, ndof), dtype=np.complex128)

            # prefactor = chara[0].real / len(chara)
            for kk in range(len(chara)):                # loop ops
                # projector += prefactor * chara[kk] * matrices[kk]
                projector += chara[kk] * matrices[kk]
            basis = fast_orth(projector, remaining_dof, int(ndof/len(idx)))
            adapted.append(basis)
            remaining_dof -= basis.shape[1]
            dimensions.append(basis.shape[1])

        adapted = np.concatenate(adapted, axis=1)
        if adapted.shape[0] != adapted.shape[1]:
            # print(adapted.shape)
            print(ii, jj, adapted.shape)
            print(dimensions)
            set_trace()
            logging.ERROR("the shape of adapted not equal")

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

    characters, paras_values, paras_symbols = get_character([0], nrot, order, family, a=cyclic._pure_trans)
    characters = np.array(characters)
    characters = characters[::2] + characters[1::2]

    ndof = 3 * len(atom.numbers)
    remaining_dof = copy.deepcopy(ndof)
    adapted = []
    dimensions = []
    for ii, chara in enumerate(characters):  # loop quantum number
        projector = np.zeros((ndof, ndof), dtype=np.complex128)
        # prefactor = chara[0].real / len(chara)
        for kk in range(len(chara)):  # loop ops
            # projector += prefactor * chara[kk] * matrices[kk]
            projector += chara[kk] * matrices[kk]
        basis = fast_orth(projector, remaining_dof, int(ndof / len(characters)))
        adapted.append(basis)
        remaining_dof -= basis.shape[1]
        dimensions.append(basis.shape[1])

    ############ q=0 adapted in pure system
    adapted = np.concatenate(adapted, axis=1)
    if adapted.shape[0] != adapted.shape[1]:
        # print(adapted.shape)
        print(ii, adapted.shape)
        print(dimensions)
        set_trace()
        logging.ERROR("the shape of adapted not equal")

    ######################################
    LR_blocks = np.load(path_LR_blocks)
    scatter_blocks = np.load(path_scatter_blocks)
    defect_indices = np.load(path_defect_indices)["defect_indices"]

    HL = LR_blocks["H00"]
    TL = LR_blocks["H01"]


    cells = max(defect_indices) + 1
    idx_scatter = np.where(defect_indices==int((cells-1)/2))[0]

    mass_L = np.diag(np.power(ase.data.atomic_masses[poscar.numbers], -1/2))
    mass_R = mass_L.copy()
    unit = 1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J

    HL = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, HL), mass_L)
    TL = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, TL), mass_L)


    # KC = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_C, KC), mass_C)
    # VLC = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, VLC), mass_C)
    # VCR = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_C, VCR), mass_R)

    HL = HL.transpose((0, 2, 1, 3)).reshape((HL.shape[0] * 3, -1))
    TL = TL.transpose((0, 2, 1, 3)).reshape((TL.shape[0] * 3, -1))
    # KC = KC.transpose((0, 2, 1, 3)).reshape((KC.shape[0] * 3, -1))
    # VLC = VLC.transpose((0, 2, 1, 3)).reshape((VLC.shape[0] * 3, -1))
    # VCR = VCR.transpose((0, 2, 1, 3)).reshape((VCR.shape[0] * 3, -1))

    aL = phonon.primitive.cell[2, 2]
    aR = aL

    # Plot the phonon spectra of both bulk leads.
    # qvec = np.linspace(0.0, 2.0 * np.pi, num=1001)
    qvec = np.linspace(-np.pi, np.pi, num=NPOINTS)
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    # Compute all the parameters of the interface
    inc_omega = np.linspace(1e-4, omegaL.max() * 1.01, num=NPOINTS, endpoint=True)


    NLp = np.empty_like(inc_omega)
    NLm = np.empty_like(inc_omega)

    HL_complex = HL.astype(complex).copy()
    TL_complex = TL.astype(complex).copy()
    for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
        en = omega * (omega + 1.0j * args.eps)

        # Build the four retarded GFs of leads extending left or right.
        inv_gLretm = decimation.inv_g00(
            HL_complex, TL_complex, omega, args.eps, args.decimation, args.maxiter
        )
        inv_gLretp = decimation.inv_g00(
            HL_complex,
            TL_complex.conj().T,
            omega,
            args.eps,
            args.decimation,
            args.maxiter,
        )
        # And the four advanced versions, i.e., their Hermitian conjugates.
        inv_gLadvm = inv_gLretm.conj().T
        inv_gLadvp = inv_gLretp.conj().T

        # Build the retarded Green's function for the interface.
        HL_pr = HL + TL.conj().T @ la.solve(inv_gLretm, TL)


        # Compute the total transmission.
        GammaL = (
            1.0j * TL.conj().T @ (la.solve(inv_gLretm, TL) - la.solve(inv_gLadvm, TL))
        )

        # Build the Bloch matrices.
        FLretp = la.solve(inv_gLretp, TL.conj().T)
        FLadvp = la.solve(inv_gLadvp, TL.conj().T)
        inv_FLretm = la.solve(inv_gLretm, TL)
        inv_FLadvm = la.solve(inv_gLadvm, TL)


        # Solve the corresponding eigenvalue equations for the leads.
        # Look for degenerate modes and orthonormalize them.

        ALretp, ULretp = orthogonalize(*la.eig(FLretp))
        ALadvp, ULadvp = orthogonalize(*la.eig(FLadvp))
        ALretm, ULretm = orthogonalize(*la.eig(inv_FLretm))
        ALadvm, ULadvm = orthogonalize(*la.eig(inv_FLadvm))

        # Find out which modes are propagating.
        mask_Lretp = np.isclose(np.abs(ALretp), 1.0, args.tolerance)
        mask_Ladvp = np.isclose(np.abs(ALadvp), 1.0, args.tolerance)
        mask_Lretm = np.isclose(np.abs(ALretm), 1.0, args.tolerance)
        mask_Ladvm = np.isclose(np.abs(ALadvm), 1.0, args.tolerance)

        # Compute the group velocity matrices.
        # yapf: disable
        VLretp = 1.j * aL * ULretp.conj().T @ TL @ (
            la.solve(inv_gLretp, TL.conj().T) -
            la.solve(inv_gLadvp, TL.conj().T)
        ) @ ULretp / 2. / omega
        VLadvp = 1.j * aL * ULadvp.conj().T @ TL @ (
            la.solve(inv_gLadvp, TL.conj().T) -
            la.solve(inv_gLretp, TL.conj().T)
        ) @ ULadvp / 2. / omega
        VLretm = -1.j * aL * ULretm.conj().T @ TL.conj().T @ (
            la.solve(inv_gLretm, TL) -
            la.solve(inv_gLadvm, TL)
        ) @ ULretm / 2. / omega
        VLadvm = -1.j * aL * ULadvm.conj().T @ TL.conj().T @ (
            la.solve(inv_gLadvm, TL) -
            la.solve(inv_gLretm, TL)
        ) @ ULadvm / 2. / omega
        # yapf: enable

        # Refine these matrices using the precomputed propagation masks.
        def refine(V, mask):
            diag = np.diag(V)
            nruter = np.zeros_like(diag)
            nruter[mask] = diag[mask].real
            return np.diag(nruter)

        VLretp = refine(VLretp, mask_Lretp)
        VLadvp = refine(VLadvp, mask_Ladvp)
        VLretm = refine(VLretm, mask_Lretm)
        VLadvm = refine(VLadvm, mask_Ladvm)

        # Set up auxiliary diagonal matrices with elements that are either
        # inverse of the previous diagonals or zero.
        def build_tilde(V):
            diag = np.diag(V)
            nruter = np.zeros_like(diag)
            indices = np.logical_not(np.isclose(np.abs(diag), 0.0))
            nruter[indices] = 1.0 / diag[indices]
            return np.diag(nruter)

        VLretp_tilde = build_tilde(VLretp)
        VLadvp_tilde = build_tilde(VLadvp)
        VLretm_tilde = build_tilde(VLretm)
        VLadvm_tilde = build_tilde(VLadvm)

        # Build the matrices used to extract the transmission of the
        # perfect leads.
        ILretp = VLretp @ VLretp_tilde
        ILadvp = VLadvp @ VLadvp_tilde
        ILretm = VLretm @ VLretm_tilde
        ILadvm = VLadvm @ VLadvm_tilde

        # Compute the number of right-propagating and left-propagating
        # channels, and perform a small sanity check.
        NLp[iomega] = np.trace(ILretp).real
        NLm[iomega] = np.trace(ILretm).real

    HL = adapted.conj().T @ HL @ adapted
    TL = adapted.conj().T @ TL @ adapted
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    # Compute all the parameters of the interface
    inc_omega = np.linspace(1e-4, omegaL.max() * 1.01, num=NPOINTS, endpoint=True)

    HL_complex = HL.astype(complex)
    TL_complex = TL.astype(complex)
    NLp_irreps = np.zeros((len(dimensions), NPOINTS))
    for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
        en = omega * (omega + 1.0j * args.eps)
        # Build the four retarded GFs of leads extending left or right.
        inv_gLretp = decimation.inv_g00(
            HL_complex, TL_complex.conj().T, omega, args.eps, args.decimation, args.maxiter,
        )
        inv_gLadvp = inv_gLretp.conj().T

        # Build the Bloch matrices.
        FLretp = la.solve(inv_gLretp, TL.conj().T)

        start = 0
        for im, dim in enumerate(dimensions):
            end = start + dim
            # Solve the corresponding eigenvalue equations for the leads.
            # Look for degenerate modes and orthonormalize them.
            ALretp, ULretp = orthogonalize(*la.eig(FLretp[start:end, start:end]))

            # Find out which modes are propagating.
            mask_Lretp = np.isclose(np.abs(ALretp), 1.0, args.tolerance)

            # Compute the group velocity matrices.
            # yapf: disable
            VLretp = 1.j * aL * ULretp.conj().T @ TL[start:end, start:end] @ (
                    la.solve(inv_gLretp, TL.conj().T)[start:end, start:end] -
                    la.solve(inv_gLadvp, TL.conj().T)[start:end, start:end]
            ) @ ULretp / 2. / omega

            start = end
            # Refine these matrices using the precomputed propagation masks.
            def refine(V, mask):
                diag = np.diag(V)
                nruter = np.zeros_like(diag)
                nruter[mask] = diag[mask].real
                return np.diag(nruter)
            VLretp = refine(VLretp, mask_Lretp)

            # Set up auxiliary diagonal matrices with elements that are either
            # inverse of the previous diagonals or zero.
            def build_tilde(V):
                diag = np.diag(V)
                nruter = np.zeros_like(diag)
                indices = np.logical_not(np.isclose(np.abs(diag), 0.0))
                nruter[indices] = 1.0 / diag[indices]
                return np.diag(nruter)
            VLretp_tilde = build_tilde(VLretp)

            ILretp = VLretp @ VLretp_tilde
            NLp_irreps[im, iomega] = np.trace(ILretp).real

    NLp_sum = NLp_irreps.sum(axis=0)
    fig, ax = plt.subplots()
    ax.plot(np.array(inc_omega), NLp, label="sum", color="grey")


    # y = np.zeros((len(inc_omega)))
    # frequencies = 2*np.pi * frequencies
    # for ii, omega in enumerate(inc_omega):
    #     counts = 0
    #     for jj, freq in enumerate(frequencies):
    #         counts += counting_y_from_xy(omega, freq)
    #     y[ii] = counts
    # ax.plot(np.array(inc_omega), y, label="Counting", color="red", linewidth=2)
    # fig1, ax1 = plt.subplots()
    # for ii, freq in enumerate(omegaL.T):
    #     ax1.plot(np.array(inc_omega), freq, color="grey", linewidth=2)


    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'yellow', 'pink']
    labels = ["|m|=0","|m|=1","|m|=2","|m|=3","|m|=4","|m|=5","|m|=6","|m|=7","|m|=8","|m|=9"]

    NLp_irreps = np.array([NLp_irreps[2], NLp_irreps[1]+NLp_irreps[3], NLp_irreps[0]+NLp_irreps[4], NLp_irreps[5]])
    for ii, freq in enumerate(NLp_irreps):
        ax.plot(np.array(inc_omega), freq, label=labels[ii], color=color[ii])

    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.legend(loc="best")
    plt.tight_layout()
    # plt.savefig(os.path.join(path_directory, "transmission_sym_adapted_pure.png"), dpi=600)
    plt.show()
