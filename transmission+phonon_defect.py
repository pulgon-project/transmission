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
import logging
import re
import glob
import sys
import os
import os.path
import math
import copy
import math
import argparse
import itertools
import collections
import functools
import pickle

import gzip
import lzma
import tqdm
import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.linalg as la
import scipy.spatial
import scipy.spatial.distance
import ase
import ase.io
import ase.data
import matplotlib
import matplotlib.pyplot as plt
import phonopy

import decimation
from ipdb import set_trace
from matplotlib.cm import ScalarMappable


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
    path_phonopy_defect = os.path.join(path_directory, "phonopy_defect.yaml")
    path_phonopy_pure = os.path.join(path_directory, "phonopy_pure.yaml")
    path_LR_blocks = os.path.join(path_directory, "pure_fc.npz")
    path_scatter_blocks = os.path.join(path_directory, "scatter_fc.npz")
    path_defect_indices = os.path.join(path_directory, "defect_indices.npz")
    path_poscar = os.path.join(path_directory, "POSCAR")
    path_savfig = os.path.join(path_directory, "phonon+transmission")

    phonon = phonopy.load(phonopy_yaml=path_phonopy_defect)
    phonon_pure = phonopy.load(phonopy_yaml=path_phonopy_pure)
    LR_blocks = np.load(path_LR_blocks)
    scatter_blocks = np.load(path_scatter_blocks)
    defect_indices = np.load(path_defect_indices)["defect_indices"]
    poscar = ase.io.read(path_poscar)

    HL = LR_blocks["H00"]
    TL = LR_blocks["H01"]

    KC = scatter_blocks["Hc"]
    VLC = scatter_blocks["Vlc"]
    VCR = scatter_blocks["Vcr"]

    cells = max(defect_indices) + 1
    idx_scatter = np.where(defect_indices==int((cells-1)/2))[0]

    mass_C = np.diag(np.power(ase.data.atomic_masses[phonon.primitive.numbers[idx_scatter]], -1/2))
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
    HR = HL.copy()
    TR = TL.copy()

    aL = phonon_pure.primitive.cell[2, 2]
    aR = aL

    # Plot the phonon spectra of both bulk leads.
    qvec = np.linspace(-np.pi, np.pi, num=NPOINTS)
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    omegaR, vgR = decimation.q2omega(HR, TR, qvec)

    # fig, ax = plt.subplots()
    # for i, f_raw in enumerate(omegaL.T):
    #     if i == 0:
    #         ax.plot(qvec, f_raw, 'o-', markersize=3, color="grey", label="1S defect")
    #     else:
    #         # ax.plot(qvec, f_raw, 'o-',markersize=3, color="grey")
    #         ax.plot(qvec, f_raw, 'o-',markersize=3)
    # plt.show()


    # Compute all the parameters of the interface
    inc_omega = np.linspace(
        1e-4, max(omegaL.max(), omegaR.max()) * 1.01, num=NPOINTS, endpoint=True
    )
    # inc_omega = np.unique(np.round(omegaL.flatten(), 1))[1::3]

    one = np.eye(KC.shape[0], dtype=np.complex128)
    trans = np.empty_like(inc_omega)
    trans_check = np.empty_like(inc_omega)

    HL_complex = HL.astype(complex)
    TL_complex = TL.astype(complex)
    HR_complex = HR.astype(complex)
    TR_complex = TR.astype(complex)
    # trans_values = np.zeros((omegaL.shape[0], omegaL.shape[1]))
    trans_values = []
    k_w = []
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
            en * np.eye(H_pr.shape[0], dtype=np.complex128) - H_pr, atol=1e-3
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
        GLRret = Gret[: HL_pr.shape[0], -HR_pr.shape[1] :]
        GRLret = Gret[-HR_pr.shape[0] :, : HL_pr.shape[1]]

        trans[iomega] = np.trace(
            GRLret @ (GRLret @ GammaL.conj().T).conj().T @ GammaR
        ).real

        # Build the Bloch matrices.
        FLretp = la.solve(inv_gLretp, TL.conj().T)
        FRretp = la.solve(inv_gRretp, TR.conj().T)
        FLadvp = la.solve(inv_gLadvp, TL.conj().T)
        FRadvp = la.solve(inv_gRadvp, TR.conj().T)
        inv_FLretm = la.solve(inv_gLretm, TL)
        inv_FRretm = la.solve(inv_gRretm, TR)
        inv_FLadvm = la.solve(inv_gLadvm, TL)
        inv_FRadvm = la.solve(inv_gRadvm, TR)

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
                    values[hi], values[hi - 1], args.tolerance
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
        mask_Lretp = np.isclose(np.abs(ALretp), 1.0, args.tolerance)
        mask_Rretp = np.isclose(np.abs(ARretp), 1.0, args.tolerance)
        mask_Ladvp = np.isclose(np.abs(ALadvp), 1.0, args.tolerance)
        mask_Radvp = np.isclose(np.abs(ARadvp), 1.0, args.tolerance)
        mask_Lretm = np.isclose(np.abs(ALretm), 1.0, args.tolerance)
        mask_Rretm = np.isclose(np.abs(ARretm), 1.0, args.tolerance)
        mask_Ladvm = np.isclose(np.abs(ALadvm), 1.0, args.tolerance)
        mask_Radvm = np.isclose(np.abs(ARadvm), 1.0, args.tolerance)

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
        )  @ URadvp / 2. / omega
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
        )  @ URadvm / 2. / omega

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

        # if not (mask_Rretp==mask_Ladvm).all():
        #     set_trace()
        #     logging.ERROR("mask_Rretp != mask_Ladvm")

        tRL = tRL[mask_Rretp, :][:, mask_Ladvm]
        # Compute the total transmission again.
        trans_check[iomega] = np.diag(tRL.conj().T @ tRL).sum().real
        trans_values.append(np.diag(tRL.conj().T @ tRL).real)
        k_w.append(np.arccos(ALadvm[mask_Ladvm].real) / aL)
        if not np.isclose(trans_check[iomega], trans[iomega], atol=1.0):
            print("Problem at omega={} rad/ps".format(inc_omega[iomega]))

    # trans_values_scaled = (trans_values - trans_values.min()) / (trans_values.max() - trans_values.min())
    # trans_values_scaled = trans_values.T
    # z = np.random.rand(NPOINTS)
    # print(z.dtype)
    # set_trace()

    qpoints_onedim = qvec / aL
    # # cmap = plt.cm.coolwarm
    for i, f_raw in enumerate(omegaL.T):
        if i == 0:
            # plt.plot(qvec, f_raw, '-', color="grey", zorder=1)
            plt.plot(qpoints_onedim, f_raw, '-', color="grey", zorder=1)
            # scatter = plt.scatter(qvec, f_raw, s=5, c=trans_values_scaled[i], cmap="coolwarm", vmin=0, vmax=1, zorder=2)
        else:
            # ax.plot(qvec, f_raw, 'o-',markersize=3, color="grey")
            # plt.plot(qvec, f_raw, '-', color="grey", zorder=1)
            plt.plot(qpoints_onedim, f_raw, '-', color="grey", zorder=1)
            # scatter = plt.scatter(qvec, f_raw, s=5, c=trans_values_scaled[i], cmap="coolwarm", vmin=0, vmax=1, zorder=2)

    for iomega, omega in enumerate(inc_omega):
        x = k_w[iomega]
        y = omega * np.ones_like(x)
        scatter = plt.scatter(x, y, s=6, c=trans_values[iomega], cmap="coolwarm", vmin=0, vmax=1, zorder=2)

    plt.colorbar(scatter, label='Values')
    plt.legend(loc="best")
    plt.savefig(path_savfig, dpi=600)
    plt.show()
