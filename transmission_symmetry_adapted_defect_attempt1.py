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
from utilities import counting_y_from_xy, get_adapted_matrix, divide_irreps, divide_over_irreps, get_adapted_matrix_multiq

matplotlib.rcParams["font.size"] = 16.0
NPOINTS = 101

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
        "-t1",
        "--rtol",
        type=float,
        default=1e-3,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
             " it a propagating mode",
    )
    parser.add_argument(
        "-t2",
        "--atol",
        type=float,
        default=5e-2,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
             " it a propagating mode",
    )
    parser.add_argument(
        "-t3",
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
    path_phonopy_defect = os.path.join(path_directory, "phonopy_defect.yaml")
    path_phonopy_pure = os.path.join(path_directory, "phonopy_pure.yaml")
    path_LR_blocks = os.path.join(path_directory, "pure_fc.npz")
    path_scatter_blocks = os.path.join(path_directory, "scatter_fc.npz")
    path_defect_indices = os.path.join(path_directory, "defect_indices.npz")
    path_poscar = os.path.join(path_directory, "POSCAR")

    ######################### projector
    phonon = phonopy.load(phonopy_yaml=path_phonopy_pure, is_compact_fc=True)

    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom = cyclic._primitive
    atom_center = find_axis_center_of_nanotube(atom)


    ################### family 4 #############
    # family = 4
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.get_rotational_symmetry_number()
    # num_irreps = nrot * 2
    # sym = []
    # tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    # sym.append(tran.affine_matrix)
    # rot = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    # sym.append(rot.affine_matrix)
    # mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    # sym.append(mirror.affine_matrix)
    ################### family 2 #############
    family = 2
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    num_irreps = nrot * 2
    sym = []
    # pg1 = obj.get_generators()  # change the order to satisfy the character table
    # sym.append(pg1[1])
    rots = SymmOp.from_rotation_and_translation(S2n(nrot), [0, 0, 0])
    sym.append(rots.affine_matrix)
    #########################################

    ops, order_ops = brute_force_generate_group_subsquent(sym)
    if len(ops) != len(order_ops):
        logging.ERROR("len(ops) != len(order)")

    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * cyclic._pure_trans
        )
        ops_car_sym.append(tmp_sym)
    matrices = get_matrices(atom_center, ops_car_sym)
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

    HL = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, HL), mass_L)
    TL = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, TL), mass_L)
    KC = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_C, KC), mass_C)
    VLC = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_L, VLC), mass_C)
    VCR = unit * np.einsum('ijlm,jk->iklm', np.einsum('ij,jkln->ikln', mass_C, VCR), mass_R)

    HL = HL.transpose((0, 2, 1, 3)).reshape((HL.shape[0] * 3, -1))
    TL = TL.transpose((0, 2, 1, 3)).reshape((TL.shape[0] * 3, -1))
    KC = KC.transpose((0, 2, 1, 3)).reshape((KC.shape[0] * 3, -1))
    VLC = VLC.transpose((0, 2, 1, 3)).reshape((VLC.shape[0] * 3, -1))
    VCR = VCR.transpose((0, 2, 1, 3)).reshape((VCR.shape[0] * 3, -1))

    # HL = adapted.conj().T @ HL @ adapted
    # TL = adapted.conj().T @ TL @ adapted
    HR = HL.copy()
    TR = TL.copy()

    aL = phonon.primitive.cell[2, 2]
    aR = aL

    # Plot the phonon spectra of both bulk leads.
    # qvec = np.linspace(0.0, 2.0 * np.pi, num=1001)
    qvec = np.linspace(-np.pi, np.pi, num=NPOINTS*10)
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    # Compute all the parameters of the interface
    inc_omega = np.linspace(1e-1, omegaL.max() * 1.01, num=NPOINTS, endpoint=True)

    trans = np.zeros_like(inc_omega)
    trans_check = np.zeros_like(inc_omega)

    HL_complex = HL.astype(complex)
    TL_complex = TL.astype(complex)
    HR_complex = HR.astype(complex)
    TR_complex = TR.astype(complex)
    matrices_prob, Irreps = [], []

    NLp_irreps = np.zeros((num_irreps, NPOINTS))  # the number of im
    for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
        # omega = 7.656578907447281
        print("---------- -----------")
        print("omega=", omega)

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
        inv_gRretm = decimation.inv_g00(
            HR_complex, TR_complex, omega, args.eps, args.decimation, args.maxiter
        )
        inv_gRretp = decimation.inv_g00(
            HR_complex,
            TR_complex.conj().T,
            omega,
            args.eps,
            args.decimation,
            args.maxiter,
        )
        # And the four advanced versions, i.e., their Hermitian conjugates.
        inv_gLadvm = inv_gLretm.conj().T
        inv_gLadvp = inv_gLretp.conj().T
        inv_gRadvm = inv_gRretm.conj().T
        inv_gRadvp = inv_gRretp.conj().T

        # Build the Bloch matrices.
        FLretp = la.solve(inv_gLretp, TL.conj().T)
        FRretp = la.solve(inv_gRretp, TR.conj().T)
        FLadvp = la.solve(inv_gLadvp, TL.conj().T)
        FRadvp = la.solve(inv_gRadvp, TR.conj().T)
        inv_FLretm = la.solve(inv_gLretm, TL)
        inv_FRretm = la.solve(inv_gRretm, TR)
        inv_FLadvm = la.solve(inv_gLadvm, TL)
        inv_FRadvm = la.solve(inv_gRadvm, TR)

        # Build the retarded Green's function for the interface.
        HL_pr = HL + TL.conj().T @ la.solve(inv_gLretm, TL)
        HR_pr = HR + TR @ la.solve(inv_gRretp, TR.conj().T)

        # yapf: disable
        H_pr = np.block([
            [HL_pr, VLC, np.zeros((HL_pr.shape[0], HR_pr.shape[1]))],
            [VLC.conj().T, KC, VCR],
            [np.zeros((HR_pr.shape[0], HL_pr.shape[1])), VCR.conj().T, HR_pr]
        ])

        # yapf: enable
        Gret = la.pinv(
            en * np.eye(H_pr.shape[0], dtype=np.complex128) - H_pr
        )
        # Compute the total transmission.
        GammaL = (
                1.0j * TL.conj().T @ (la.solve(inv_gLretm, TL) - la.solve(inv_gLadvm, TL))
        )
        GammaR = (
                1.0j
                * TR
                @ (la.solve(inv_gRretp, TR.conj().T) - la.solve(inv_gRadvp, TR.conj().T))
        )
        GLRret = Gret[: HL_pr.shape[0], -HR_pr.shape[1]:]
        GRLret = Gret[-HR_pr.shape[0]:, : HL_pr.shape[1]]

        trans[iomega] = np.trace(
            GRLret @ (GRLret @ GammaL.conj().T).conj().T @ GammaR
        ).real

        values, vectors = la.eig(inv_FLadvm)
        mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)
        order_val = np.lexsort((np.angle(values), 1 * (~mask)))
        values = values[order_val]
        vectors = vectors[:, order_val]
        mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)
        lo = 0
        hi = 1
        groups = []
        while True:
            if hi >= vectors.shape[1] or not np.isclose(
                    values[hi], values[hi - 1], rtol=args.rtol, atol=args.atol,
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

        # mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)
        irreps = []
        if mask.sum() != 0:  # not all False
            # idx_mask_end = np.where(mask)[0][-1]
            # k_w = np.abs(np.angle(values[mask])) / aL
            k_w = np.arccos(values[mask].real) / aL
            k_adapteds = np.unique(k_w)

            adapteds, dimensions = [], []
            for qp in k_adapteds:
                DictParams = {"qpoints": qp, "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13
                adapted, dimension = get_adapted_matrix(DictParams, num_atoms, matrices)

                adapteds.append(adapted)
                dimensions.append(dimension)

        # adapted0, dim0 = get_adapted_matrix(0, nrot, order_ops, family, aL, num_atoms, matrices)
        def orthogonalize(values, vectors, adapteds, k_adapteds, dimensions):
            mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)
            order_val = np.lexsort((np.angle(values), 1 * (~mask)))
            values = values[order_val]
            vectors = vectors[:, order_val]
            mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)

            lo = 0
            hi = 1
            groups = []
            while True:
                if hi >= vectors.shape[1] or not np.isclose(
                        values[hi], values[hi - 1], rtol=args.rtol, atol=args.atol,
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
                    vectors[:,lo:hi] = la.orth(vectors[:,lo:hi])

            # mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)
            indices = np.arange(len(mask))
            group_masks = []
            for g in groups:
                lo, hi = g
                group_masks.append(mask & (indices >= lo) & (indices < hi))

            irreps = []
            for i_m, m in enumerate(group_masks):
                degeneracy = m.sum()
                if degeneracy == 0:
                    continue

                # k_w_group = np.abs(np.angle(values[m][0])) / aL
                k_w_group = np.arccos(values[m][0].real) / aL

                # print("k_w_group: ", k_w_group)
                # k_w_group = np.arccos(values[m][0].real) / aL

                try:
                    k_w_group_indice = np.where(np.isclose(k_w_group, k_adapteds))[0].item()
                except:
                    set_trace()
                    logging.ERROR("No correspond k indices")
                basis, dims = adapteds[k_w_group_indice], dimensions[k_w_group_indice]
                # basis, dims = adapted0, dim0

                group_vectors = vectors[:, m]
                try:
                    # tmp = divide_irreps(group_vectors.T, basis, dims).sum(axis=0)
                    adapted_vecs = divide_over_irreps(group_vectors, basis, dims)
                except:
                    res = divide_irreps(group_vectors.T, basis, dims)
                    set_trace()

                # res = check_same_space(group_vectors, adapted_vecs)
                # print(f"Group {i_m}: {group_vectors.shape[1]} modes")
                tmp_vec = []
                for i_ir, v in enumerate(adapted_vecs):
                    if v.shape[1] > 0:
                        tmp_vec.append(v)
                        irreps.extend(np.repeat(i_ir, v.shape[1]))
                        # print(f"\t- {v.shape[1]} modes in irrep #{i_ir+1}")

                tmp_vec = np.concatenate(tmp_vec, axis=1)
                vectors[:, m] = tmp_vec
            return values, vectors, mask, irreps


        # # Solve the corresponding eigenvalue equations for the leads.
        # # Look for degenerate modes and orthonormalize them.
        ALadvm, ULadvm, mask_Ladvm, tmp_irreps = orthogonalize(*la.eig(inv_FLadvm), adapteds, k_adapteds, dimensions)
        ALretp, ULretp, mask_Lretp, _ = orthogonalize(*la.eig(FLretp), adapteds, k_adapteds, dimensions)
        ARretp, URretp, mask_Rretp, _ = orthogonalize(*la.eig(FRretp), adapteds, k_adapteds, dimensions)
        ALadvp, ULadvp, mask_Ladvp, _ = orthogonalize(*la.eig(FLadvp), adapteds, k_adapteds, dimensions)
        ARadvp, URadvp, mask_Radvp, _ = orthogonalize(*la.eig(FRadvp), adapteds, k_adapteds, dimensions)

        ALretm, ULretm, mask_Lretm, _ = orthogonalize(*la.eig(inv_FLretm), adapteds, k_adapteds, dimensions)
        ARretm, URretm, mask_Rretm, _ = orthogonalize(*la.eig(inv_FRretm), adapteds, k_adapteds, dimensions)
        ARadvm, URadvm, mask_Radvm, _ = orthogonalize(*la.eig(inv_FRadvm), adapteds, k_adapteds, dimensions)


        # Compute the group velocity matrices.
        # yapf: disable
        VLretp = 1.j * aL * ULretp.conj().T @ TL @ (
                la.solve(inv_gLretp, TL.conj().T) -
                la.solve(inv_gLadvp, TL.conj().T)
        ) @ ULretp / 2. / omega
        VRretp = 1.j * aR * URretp.conj().T @ TR @ (
                la.solve(inv_gRretp, TR.conj().T) -
                la.solve(inv_gRadvp, TR.conj().T)
        ) @ URretp / 2. / omega
        VLadvp = 1.j * aL * ULadvp.conj().T @ TL @ (
                la.solve(inv_gLadvp, TL.conj().T) -
                la.solve(inv_gLretp, TL.conj().T)
        ) @ ULadvp / 2. / omega
        VRadvp = 1.j * aR * URadvp.conj().T @ TR @ (
                la.solve(inv_gRadvp, TR.conj().T) -
                la.solve(inv_gRretp, TR.conj().T)
        ) @ URadvp / 2. / omega
        VLretm = -1.j * aL * ULretm.conj().T @ TL.conj().T @ (
                la.solve(inv_gLretm, TL) -
                la.solve(inv_gLadvm, TL)
        ) @ ULretm / 2. / omega
        VRretm = -1.j * aR * URretm.conj().T @ TR.conj().T @ (
                la.solve(inv_gRretm, TR) -
                la.solve(inv_gRadvm, TR)
        ) @ URretm / 2. / omega
        VLadvm = -1.j * aL * ULadvm.conj().T @ TL.conj().T @ (
                la.solve(inv_gLadvm, TL) -
                la.solve(inv_gLretm, TL)
        ) @ ULadvm / 2. / omega
        VRadvm = -1.j * aR * URadvm.conj().T @ TR.conj().T @ (
                la.solve(inv_gRadvm, TR) -
                la.solve(inv_gRretm, TR)
        ) @ URadvm / 2. / omega


        # yapf: enable
        # Refine these matrices using the precomputed propagation masks.
        def refine(V, mask):
            diag = np.diag(V)
            nruter = np.zeros_like(diag)
            nruter[mask] = diag[mask].real
            return np.diag(nruter)


        VLretp = refine(VLretp, mask_Lretp)
        VRretp = refine(VRretp, mask_Rretp)
        VLadvp = refine(VLadvp, mask_Ladvp)
        VRadvp = refine(VRadvp, mask_Radvp)
        VLretm = refine(VLretm, mask_Lretm)
        VRretm = refine(VRretm, mask_Rretm)
        VLadvm = refine(VLadvm, mask_Ladvm)
        VRadvm = refine(VRadvm, mask_Radvm)


        # Set up auxiliary diagonal matrices with elements that are either
        # inverse of the previous diagonals or zero.
        def build_tilde(V):
            diag = np.diag(V)
            nruter = np.zeros_like(diag)
            indices = np.logical_not(np.isclose(np.abs(diag), 0.0))
            nruter[indices] = 1.0 / diag[indices]
            return np.diag(nruter)


        VLretp_tilde = build_tilde(VLretp)
        VRretp_tilde = build_tilde(VRretp)
        VLadvp_tilde = build_tilde(VLadvp)
        VRadvp_tilde = build_tilde(VRadvp)
        VLretm_tilde = build_tilde(VLretm)
        VRretm_tilde = build_tilde(VRretm)
        VLadvm_tilde = build_tilde(VLadvm)
        VRadvm_tilde = build_tilde(VRadvm)

        # Build the matrices used to extract the transmission of the
        # perfect leads.
        ILretp = VLretp @ VLretp_tilde
        IRretp = VRretp @ VRretp_tilde
        ILadvp = VLadvp @ VLadvp_tilde
        IRadvp = VRadvp @ VRadvp_tilde
        ILretm = VLretm @ VLretm_tilde
        IRretm = VRretm @ VRretm_tilde
        ILadvm = VLadvm @ VLadvm_tilde
        IRadvm = VRadvm @ VRadvm_tilde

        VRretp12 = np.sqrt(VRretp)
        VLadvm12 = np.sqrt(VLadvm)
        VLretm12 = np.sqrt(VLretm)
        VRadvp12 = np.sqrt(VRadvp)

        tRL = (
                2.0j
                * omega
                * (
                        VRretp12
                        @ la.solve(URretp, GRLret)
                        @ la.solve(ULadvm.conj().T, VLadvm12)
                        / np.sqrt(aR * aL)
                )
        )
        tLR = (
                2.0j
                * omega
                * (
                        VLretm12
                        @ la.solve(ULretm, GLRret)
                        @ la.solve(URadvp.conj().T, VRadvp12)
                        / np.sqrt(aR * aL)
                )
        )

        #  Discard evanescent modes.
        tRL = tRL[mask_Rretp, :][:, mask_Ladvm]
        trans_modes = np.diag(tRL.conj().T @ tRL).real
        trans_check[iomega] = trans_modes.sum()



        matrices_prob.append(np.diag(trans_modes))
        Irreps.append(np.array(tmp_irreps) - 2)
        for im, tras in enumerate(trans_modes):
            NLp_irreps[tmp_irreps[im], iomega] += tras


    NLp_sum = NLp_irreps.sum(axis=0)

    # if True and NPOINTS == 50:
    #     fsize = matplotlib.rcParams["font.size"]
    #     fig, axs = plt.subplots(5, 10, figsize=(16, 10))
    #     # fig, axs = plt.subplots(5, 10, figsize=(18, 12))
    #     for i in range(5):
    #         for j in range(10):
    #             k = 10 * i + j
    #             if matrices_prob[k].size > 0:
    #                 im = axs[i, j].matshow(matrices_prob[k], vmin=0.0, vmax=1.0)
    #                 # axs[i, j].set_xticks(np.arange(len(Irreps[k])), Irreps[k])
    #                 # axs[i, j].set_yticks(np.arange(len(Irreps[k])), Irreps[k])
    #                 axs[i, j].set_xticks(np.arange(len(Irreps[k])))
    #                 axs[i, j].set_xticklabels(Irreps[k])
    #                 axs[i, j].set_yticks(np.arange(len(Irreps[k])))
    #                 axs[i, j].set_yticklabels(Irreps[k])
    #                 axs[i, j].tick_params(axis='x', labelsize=8)
    #                 axs[i, j].tick_params(axis='y', labelsize=8)
    #             else:
    #                 axs[i, j].axis("off")
    #             axs[i, j].text(
    #                 0.5,
    #                 0.5,
    #                 r"${0:.2f}$".format(inc_omega[k]),
    #                 horizontalalignment="center",
    #                 verticalalignment="center",
    #                 transform=axs[i, j].transAxes,
    #                 fontdict=dict(size=fsize / 1.2, color="red", weight="bold"),
    #             )
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=1e-3, wspace=1e-3, right=0.9)
    # cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.9])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.savefig(os.path.join(path_directory, "section.png"), dpi=500)


    fig, axs = plt.subplots(figsize=(8, 6))
    plt.plot(np.array(inc_omega), trans_check, label="modes_sum", color="yellow")
    plt.plot(np.array(inc_omega), trans, label="Caroli", color="grey")
    plt.plot(np.array(inc_omega), NLp_sum, label="Irreps_sum", color="pink")

    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'yellow', 'pink']
    labels = ["|m|=0", "|m|=1", "|m|=2", "|m|=3", "|m|=4", "|m|=5", "|m|=6", "|m|=7", "|m|=8", "|m|=9"]

    NLp_irreps = np.array([NLp_irreps[2], NLp_irreps[1] + NLp_irreps[3], NLp_irreps[0] + NLp_irreps[4], NLp_irreps[5]])
    for ii, freq in enumerate(NLp_irreps):
        plt.plot(np.array(inc_omega), freq, label=labels[ii], color=color[ii])

    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.xlabel("$\omega\;(\mathrm{rad/ps})$", fontsize=12)
    plt.ylabel(r"$T(\omega)$", fontsize=12)

    # plt.savefig(os.path.join(path_directory, "transmission_sym_adapted_defect.png"), dpi=600)
    plt.show()


