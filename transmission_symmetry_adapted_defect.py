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
from utilities import counting_y_from_xy, get_adapted_matrix, devide_irreps, combination_paras, refine_qpoints


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
        default=4e-3,
        help="if a mode's eigenvalue has modulus > 1 - tolerance, consider"
        " it a propagating mode",
    )
    parser.add_argument(
        "-t3",
        "--means_tol",
        type=float,
        default=7e-1,
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

    k_start = -np.pi
    k_end = np.pi

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NPOINTS
    )
    qpoints = qpoints[0]

    qpoints_1dim = np.linspace(k_start, k_end, num=NPOINTS, endpoint=k_end)
    a = cyclic._pure_trans
    qpoints_1dim = qpoints_1dim / a

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()

    sym = []
    # tran = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 1])
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    # pg1 = obj.get_generators()    # change the order to satisfy the character table
    # sym.append(pg1[1])
    S12 = np.array([[0.8660254, -0.5, 0., 0.],
                    [0.5, 0.8660254, 0., 0.],
                    [0., 0., -1., 0.],
                    [0., 0., 0., 1.]])
    sym.append(S12)
    # set_trace()
    # mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    # sym.append(mirror.affine_matrix)


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

    # set_trace()

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
    idx_scatter = np.where(defect_indices==int((cells-1)/2))[0]

    mass_C = np.diag(np.power(ase.data.atomic_masses[phonon_defect.primitive.numbers[idx_scatter]], -1/2))
    mass_L = np.diag(np.power(ase.data.atomic_masses[poscar.numbers], -1/2))
    mass_R = mass_L.copy()
    unit = 1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J

    HL = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, HL), mass_L)
    TL = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, TL), mass_L)
    KC = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_C, KC), mass_C)
    VLC = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, VLC), mass_C)
    VCR = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_C, VCR), mass_R)

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
    qvec = np.linspace(-np.pi, np.pi, num=NPOINTS)
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    # Compute all the parameters of the interface
    inc_omega = np.linspace(1e-1, omegaL.max() * 1.01, num=NPOINTS, endpoint=True)

    trans = np.zeros_like(inc_omega)
    trans_check = np.zeros_like(inc_omega)
    # NLp = np.empty_like(inc_omega)
    # NRp = np.empty_like(inc_omega)
    # NLm = np.empty_like(inc_omega)
    # NRm = np.empty_like(inc_omega)

    HL_complex = HL.astype(complex)
    TL_complex = TL.astype(complex)
    HR_complex = HR.astype(complex)
    TR_complex = TR.astype(complex)
    matrices_prob, Irreps = [], []
    NLp_irreps = np.zeros((6, NPOINTS))   # the number of im
    for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
        # omega=56.77013783915039

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
                        np.angle(values[hi]) , np.angle(values[hi-1]) , rtol=args.rtol, atol=args.atol,
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
                    vectors[:, lo:hi] = la.orth(vectors[:, lo:hi], rcond=1e-3)
            return values, vectors

        # Solve the corresponding eigenvalue equations for the leads.
        # Look for degenerate modes and orthonormalize them.
        ALretp, ULretp = orthogonalize(*la.eig(FLretp))
        ARretp, URretp = orthogonalize(*la.eig(FRretp))
        ALadvp, ULadvp = orthogonalize(*la.eig(FLadvp))
        ARadvp, URadvp = orthogonalize(*la.eig(FRadvp))
        ALretm, ULretm = orthogonalize(*la.eig(inv_FLretm))
        ARretm, URretm = orthogonalize(*la.eig(inv_FRretm))
        ALadvm, ULadvm = orthogonalize(*la.eig(inv_FLadvm))
        ARadvm, URadvm = orthogonalize(*la.eig(inv_FRadvm))

        # Find out which modes are propagating.
        mask_Lretp = np.isclose(np.abs(ALretp), 1.0, args.rtol)
        mask_Rretp = np.isclose(np.abs(ARretp), 1.0, args.rtol)
        mask_Ladvp = np.isclose(np.abs(ALadvp), 1.0, args.rtol)
        mask_Radvp = np.isclose(np.abs(ARadvp), 1.0, args.rtol)
        mask_Lretm = np.isclose(np.abs(ALretm), 1.0, args.rtol)
        mask_Rretm = np.isclose(np.abs(ARretm), 1.0, args.rtol)
        mask_Ladvm = np.isclose(np.abs(ALadvm), 1.0, args.rtol)
        mask_Radvm = np.isclose(np.abs(ARadvm), 1.0, args.rtol)

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

        # Compute the matrices required for the transmission of the complete
        # system.
        QL = (
                en * np.eye(HL.shape[0], dtype=np.complex128)
                - HL
                - TL.conj().T @ la.solve(inv_gLretm, TL)
                - TL @ la.solve(inv_gLretp, TL.conj().T)
        )
        QR = (
                en * np.eye(HR.shape[0], dtype=np.complex128)
                - HR
                - TR.conj().T @ la.solve(inv_gRretm, TR)
                - TR @ la.solve(inv_gRretp, TR.conj().T)
        )

        #  Discard evanescent modes.
        tRL = tRL[mask_Rretp, :][:, mask_Ladvm]
        trans_modes = np.diag(tRL.conj().T @ tRL)

        # tRL2 = tRL[mask_Rretp, :][:, mask_Ladvm][:2,:2]
        # trans_modes2 = np.diag(tRL2.conj().T @ tRL2)
        trans_check[iomega] = trans_modes.sum().real

        eigvec_modes = ULadvm[:,mask_Ladvm]
        if eigvec_modes.ndim>1:
            eigvec_modes = eigvec_modes.T
        print("---------- -----------")
        print("omega=", omega)
        k_w = np.abs(np.angle(ALadvm[mask_Ladvm]) /aL)

        # set_trace()
        # k_w, order_q = refine_qpoints(k_w, tol=1e-4)
        # eigvec_modes = eigvec_modes[order_q]
        # tRL = tRL[order_q, :][:, order_q]
        # k_w = np.arccos(ALadvm[mask_Ladvm].real) / aL
        # k_w = np.angle(ALadvm[mask_Ladvm].real) / aL

        k_w_unique = np.unique(k_w)
        print("k_w: ", k_w)
        print("k_w_unique: ", k_w_unique)
        # trs1 = np.diag(tRL.conj().T @ tRL).real
        # trs2 = np.abs(np.diag(tRL.conj().T @ tRL))
        tmp_transmission_prob, tmp_irreps, tmp_idx = [], [], []
        for ik, tmp_k in enumerate(k_w_unique):
            print("k=", tmp_k)
            # if np.isclose(tmp_k, 0.765102065473848) and np.isclose(omega, 20.387072865742017):
            #     set_trace()
            adapted, dimensions = get_adapted_matrix([tmp_k], nrot, order, 2, aL, num_atoms, matrices)

            itp1 = (k_w==tmp_k)
            itp2 = np.where(k_w==tmp_k)[0]
            num_mode = itp1.sum()
            if num_mode==1:       # non-degenerated q
                tmp_idx.extend(itp2)
                vec = eigvec_modes[itp2.item()]
                itp3 = devide_irreps(vec,adapted,dimensions)
                itp4 = np.where(itp3>args.means_tol)[0]
                if len(itp4)!=1:     # correspond to more than one Irreps, it means it should be degenerated with the next q
                    k_w[itp2] = k_w_unique[ik + 1]
                else:
                    tmp_irreps.extend(itp4)
                    probabilities = trans_modes[itp2].real
                    tmp_transmission_prob.extend(probabilities)
            else:             # degenerated q
                tmp_idx.extend(itp2)
                vectors = eigvec_modes[itp2]
                
                means = (devide_irreps(vectors, adapted, dimensions)>args.means_tol)
                means_unique = np.unique(means,axis=0)
                if means_unique.shape[0]==1:           # degenerated q correspond to the same Irreps basis
                    irps, paras, _ = combination_paras(vectors, adapted, means, dimensions, tol=args.means_tol)
                    paras_abs = np.abs(paras) ** 2

                    probabilities = trans_modes[itp2].real  @ paras_abs
                    tmp_irreps.extend(irps)
                    tmp_transmission_prob.extend(probabilities)
                    if len(irps) != len(probabilities):
                        if (len(probabilities)-len(irps))==1:
                            paras_abs = np.vstack((paras_abs, paras_last))
                            paras_last = (1 - paras_abs.sum(axis=0))

                            irps_last = \
                            np.where(devide_irreps(paras_last @ vectors, adapted, dimensions) > args.means_tol)[0]
                            tmp_irreps.extend(irps_last)
                            probabilities_last = paras_last @ trans_modes[itp2].real
                            tmp_transmission_prob.extend(probabilities_last)
                        else:
                            set_trace()
                            logging.Error("the num of irps and probability not equal")
                else:                # degenerated q correspond to the different Irreps basis, separate these q based on Irreps basis
                    for mea in means_unique:
                        tmp1 = (means==mea).all(axis=1) 
                        tmp_itp = np.where(tmp1)[0]
                        if len(tmp_itp)==1:
                            tmp_irreps.extend(np.where(means[tmp_itp].flatten())[0])
                            probabilities = trans_modes[itp2][tmp_itp].real
                            if len(np.where(means[tmp_itp].flatten())[0]) != len(probabilities):
                                set_trace()
                                logging.Error("one mode correspond to lots of probabilities")
                            tmp_transmission_prob.append(probabilities.item())
                        else:
                            irps, paras, _ = combination_paras(vectors[tmp_itp], adapted, means[tmp_itp], dimensions, tol=args.means_tol)

                            tmp_irreps.extend(irps)
                            if paras.ndim ==1:
                                probabilities = trans_modes[itp2][tmp_itp].real
                            else:
                                probabilities = trans_modes[itp2][tmp_itp].real @ (np.abs(paras)**2)
                            tmp_transmission_prob.extend(probabilities)

                            if len(irps) != len(probabilities):
                                if (len(probabilities)-len(irps))==1:
                                    paras_abs = np.abs(paras) ** 2
                                    paras_last = (1 - paras_abs.sum(axis=0))
                                    paras_abs = np.vstack((paras_abs, paras_last))

                                    irps_last = \
                                        np.where(
                                            devide_irreps(paras_last @ vectors, adapted, dimensions) > args.means_tol)[
                                            0]
                                    probabilities_last = irps_last @ trans_modes[itp2][tmp_itp].real
                                    tmp_irreps.extend(irps_last)
                                    tmp_transmission_prob.extend(probabilities_last)
                                else:
                                    set_trace()
                                    logging.Error("number of modes - num of Irreps > 1")

        matrices_prob.append(np.diag(tmp_transmission_prob))
        Irreps.append(np.array(tmp_irreps) - 2)
        for im, tras in enumerate(tmp_transmission_prob):
            try:
                NLp_irreps[tmp_irreps[im], iomega] += tras
            except:
                set_trace()
    NLp_sum = NLp_irreps.sum(axis=0)
    
    if True and NPOINTS == 50:
        fsize = matplotlib.rcParams["font.size"]
        # fig, axs = plt.subplots(5, 10, figsize=(16, 10))
        fig, axs = plt.subplots(5, 10, figsize=(24,16))
        for i in range(5):
            for j in range(10):
                k = 10 * i + j
                if matrices_prob[k].size > 0:
                    im = axs[i, j].matshow(matrices_prob[k], vmin=0.0, vmax=1.0)
                    # axs[i, j].set_xticks(np.arange(len(Irreps[k])), Irreps[k])
                    # axs[i, j].set_yticks(np.arange(len(Irreps[k])), Irreps[k])
                    axs[i, j].set_xticks(np.arange(len(Irreps[k])))
                    axs[i, j].set_xticklabels(Irreps[k])
                    axs[i, j].set_yticks(np.arange(len(Irreps[k])))
                    axs[i, j].set_yticklabels(Irreps[k])
                else:
                    axs[i, j].axis("off")
                axs[i, j].text(
                    0.5,
                    0.5,
                    r"${0:.2f}$".format(inc_omega[k]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axs[i, j].transAxes,
                    fontdict=dict(size=fsize / 1.2, color="red", weight="bold"),
                )
    plt.tight_layout()      
    plt.subplots_adjust(hspace=1e-3, wspace=1e-3, right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(os.path.join(path_directory, "section.png"), dpi=500)

    fig, ax = plt.subplots()
    plt.plot(np.array(inc_omega), trans_check, label="modes_sum", color="yellow")
    plt.plot(np.array(inc_omega), trans, label="Caroli", color="grey")
    plt.plot(np.array(inc_omega), NLp_sum, label="Irreps_sum", color="pink")

    # fig1, ax1 = plt.subplots()
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'yellow', 'pink']
    labels = ["|m|=0","|m|=1","|m|=2","|m|=3","|m|=4","|m|=5","|m|=6","|m|=7","|m|=8","|m|=9"]

    NLp_irreps = np.array([NLp_irreps[2],NLp_irreps[1]+NLp_irreps[3],NLp_irreps[0]+NLp_irreps[4],NLp_irreps[5]])
    for ii, freq in enumerate(NLp_irreps):
        plt.plot(np.array(inc_omega), freq, label=labels[ii], color=color[ii])

    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
    plt.ylabel(r"$T(\omega)$")

    plt.savefig(os.path.join(path_directory, "transmission_sym_adapted_defect.png"), dpi=600)
    plt.show()
