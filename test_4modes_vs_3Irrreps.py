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
from utilities import counting_y_from_xy, get_adapted_matrix, devide_irreps, divide_irreps2, combination_paras, refine_qpoints

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
        default=5e-3,
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

    k_start = -np.pi
    k_end = np.pi

    path = [[[0, 0, k_start / 2 / np.pi], [0, 0, k_end / 2 / np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NPOINTS
    )
    qpoints = qpoints[0]

    qpoints_1dim = np.linspace(k_start, k_end, num=NPOINTS, endpoint=k_end)
    a = cyclic._pure_trans
    qpoints_1dim = qpoints_1dim / a

    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()

    family = 2
    sym = []
    tran = SymmOp.from_rotation_and_translation(Cn(2 * nrot), [0, 0, 1 / 2])
    # pg1 = obj.get_generators()    # change the order to satisfy the character table
    # sym.append(pg1[1])
    S12 = np.array([[0.8660254, -0.5, 0., 0.],
                    [0.5, 0.8660254, 0., 0.],
                    [0., 0., -1., 0.],
                    [0., 0., 0., 1.]])
    sym.append(S12)

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
    NLp_irreps = np.zeros((6, NPOINTS))  # the number of im


    omega=56.77013783915039
    inc_omega = [omega]
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

        def orthogonalize(values, vectors, nrot, order_character, family, aL, num_atoms, matrices):
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
                        np.angle(values[hi]), np.angle(values[hi - 1]), rtol=args.rtol, atol=args.atol,
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

            mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)

            irreps = []
            if mask.sum()!=0:     # not all False
                idx_mask_end = np.where(mask)[0][-1]

                k_w = np.abs(np.angle(values[mask]) / aL)
                k_unique = np.unique(k_w)
                adapted, dimensions = get_adapted_matrix(k_unique, nrot, order_character, family, aL, num_atoms, matrices)

                for g in groups:
                    lo, hi = g

                    if hi <= idx_mask_end + 1:
                        tmp_value = np.abs(np.angle(values[lo]) / aL)
                        tmp_itp = np.where(k_unique == tmp_value)[0]

                        if hi == lo + 1:
                            means1 = divide_irreps2(vectors[:, lo:hi].T[0], adapted[tmp_itp.item()], dimensions[tmp_itp.item()])
                            tmp_irreps = means1 > args.means_tol
                            
                            if tmp_irreps.sum()==1:
                                irreps.extend(np.where(tmp_irreps)[0])
                            else:
                                set_trace()
                                logging.ERROR('irreps more than one')
                        elif hi > lo + 1:
                            # if tmp_itp.size == 0:
                            #     vectors[:, lo:hi] = la.orth(vectors[:, lo:hi])
                            if tmp_itp.size == 1:
                                means1 = divide_irreps2(vectors[:, lo:hi].T, adapted[tmp_itp.item()], dimensions[tmp_itp.item()])
                                dim = dimensions[tmp_itp.item()][0]

                                Irreps_num = np.round(means1.sum(axis=0)).astype(np.int32)
                                if np.abs(Irreps_num - means1.sum(axis=0)).mean() < args.means_tol :
                                    new_vec = []
                                    for ir, reps in enumerate(Irreps_num):
                                        if reps == 1:
                                            tmp_itp1 = ir * dim
                                            tmp_itp2 = ir * dim + dim

                                            tmp1 = ((vectors[:, lo:hi].sum(axis=1) @ adapted[tmp_itp.item()]))
                                            tmp2 = tmp1[tmp_itp1:tmp_itp2]
                                            # tmp3 = (tmp2[np.newaxis,:] * la.pinv(adapted[tmp_itp.item()][:, tmp_itp1:tmp_itp2]).T)
                                            tmp3 = tmp2[np.newaxis,:] * adapted[tmp_itp.item()][:, tmp_itp1:tmp_itp2]
                                            tmp_vec = la.orth(tmp3.sum(axis=1)[:,np.newaxis])
                                            tmp_means = divide_irreps2(tmp_vec.T[0], adapted[tmp_itp.item()], dimensions[tmp_itp.item()])

                                            new_vec.append(tmp_vec)
                                        elif reps >=2:
                                            tmp_itp1 = ir * dim
                                            tmp_itp2 = ir * dim + dim
                                            
                                            tmp1 = (vectors[:, lo:hi].sum(axis=1) @ adapted[tmp_itp.item()])
                                            tmp2 = tmp1[tmp_itp1:tmp_itp2]
                                            # tmp3 = (tmp2[np.newaxis,:] * la.pinv(adapted[tmp_itp.item()][:, tmp_itp1:tmp_itp2]).T)
                                            tmp3 = tmp2[np.newaxis,:] * adapted[tmp_itp.item()][:, tmp_itp1:tmp_itp2]
                                            # tmp_vec = tmp3.sum(axis=1)[:,np.newaxis]
                                            tmp_vec = la.orth(tmp3[:,:reps])

                                            tmp_means = divide_irreps2(tmp_vec.T, adapted[tmp_itp.item()], dimensions[tmp_itp.item()])

                                            new_vec.append(tmp_vec)
                                    new_vec = np.concatenate(new_vec, axis=1)
                                    vectors[:, lo:hi] = new_vec
                                    means2 = divide_irreps2(new_vec.T, adapted[tmp_itp.item()], dimensions[tmp_itp.item()])
                                    # set_trace()
                                    tmp_irreps = means2 > args.means_tol
                                    if (tmp_irreps.sum(axis=1)==1).all():
                                        irreps.extend(np.where(tmp_irreps)[1])
                                    else:
                                        set_trace()
                                        logging.ERROR("Each vector can only corrrespond to one irreps")
                                else:
                                    set_trace()
                                    logging.ERROR("It's not close to an integer")

                            else:
                                set_trace()
                                logging.ERROR("No corresponding k within mask")
            return values, vectors, mask, irreps


        # # Solve the corresponding eigenvalue equations for the leads.
        # # Look for degenerate modes and orthonormalize them.

        ALadvm, ULadvm, mask_Ladvm, irreps = orthogonalize(*la.eig(inv_FLadvm), nrot, order, family, aL, num_atoms, matrices)
        ALretp, ULretp, mask_Lretp, _ = orthogonalize(*la.eig(FLretp), nrot, order, family, aL, num_atoms, matrices)
        ARretp, URretp, mask_Rretp, _ = orthogonalize(*la.eig(FRretp), nrot, order, family, aL, num_atoms, matrices)
        ALadvp, ULadvp, mask_Ladvp, _ = orthogonalize(*la.eig(FLadvp), nrot, order, family, aL, num_atoms, matrices)
        ARadvp, URadvp, mask_Radvp, _ = orthogonalize(*la.eig(FRadvp), nrot, order, family, aL, num_atoms, matrices)
        ALretm, ULretm, mask_Lretm, _ = orthogonalize(*la.eig(inv_FLretm), nrot, order, family, aL, num_atoms, matrices)
        ARretm, URretm, mask_Rretm, _ = orthogonalize(*la.eig(inv_FRretm), nrot, order, family, aL, num_atoms, matrices)
        ARadvm, URadvm, mask_Radvm, _ = orthogonalize(*la.eig(inv_FRadvm), nrot, order, family, aL, num_atoms, matrices)


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


        set_trace()

        # tRL2 = tRL[mask_Rretp, :][:, mask_Ladvm][:2,:2]
        # trans_modes2 = np.diag(tRL2.conj().T @ tRL2)
        trans_check[iomega] = trans_modes.sum().real

        eigvec_modes = ULadvm[:, mask_Ladvm]
        if eigvec_modes.ndim > 1:
            eigvec_modes = eigvec_modes.T
        print("---------- -----------")
        print("omega=", omega)
        k_w = np.abs(np.angle(ALadvm[mask_Ladvm]) / aL)
        # k_w_unique = np.unique(k_w)

        print("k_w: ", k_w)
        print("k=", test_k)

        tmp_transmission_prob, tmp_irreps, tmp_idx = [], [], []


        itp1 = (k_w == test_k)
        itp2 = np.where(k_w == test_k)[0]
        num_mode = itp1.sum()

        tmp_idx.extend(itp2)
        vectors = eigvec_modes[itp2]
        means = (devide_irreps(vectors, adapted, dimensions) > args.means_tol)


        set_trace()


        means_unique = np.unique(means, axis=0)
        if means_unique.shape[0] == 1:  # degenerated q correspond to the same Irreps basis
            irps, paras, _ = combination_paras(vectors, adapted, means, dimensions, tol=args.means_tol)
            tmp_irreps.extend(irps)
            probabilities = trans_modes[itp2].real
            if len(irps) != len(probabilities):
                if (len(probabilities) - len(irps)) == 1:
                    paras_abs = np.abs(paras) ** 2
                    paras_last = (1-paras_abs.sum(axis=0))
                    paras_abs = np.vstack((paras_abs, paras_last))

                    irps_last = np.where(devide_irreps(paras_last @ vectors, adapted, dimensions) > args.means_tol)[0]
                    tmp_irreps.extend(irps_last)
                    probabilities_new = paras_abs @ probabilities
                else:
                    set_trace()
                    logging.Error("the num of irps and probability not equal")
            tmp_transmission_prob.extend(probabilities_new)

    print("The Irreps of new modes is: ", tmp_irreps)
    print("The transmission probibalities of new modes is: ", tmp_transmission_prob)
    print("The sum of transmission in original modes:",  probabilities.sum())
    print("The sum of transmission in new modes:",  probabilities_new.sum())
