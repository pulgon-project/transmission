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

import os.path
import argparse

import tqdm
import numpy as np

import scipy.linalg as la
import ase.io
import ase.data
import matplotlib
import matplotlib.pyplot as plt
import phonopy
import sys

import decimation
import time

import multiprocessing
from functools import partial
from ipdb import set_trace


matplotlib.rcParams["font.size"] = 16.0

# NOTE: direction is still partially hardcoded to z.

TIME_KEYS = [
    "q2omega",
    "decimation",
    "build_Gammas",
    "build_Bloch",
    "orthogonalize",
    "compute_group_velocities",
    "refine_group_velocities",
    "build_tilde",
    "build_channels",
    "compute_transmission",
]


def timed_decimation_inv_g00(HL, TL, omega, eps, decimation_param, maxiter):
    t1 = time.time()
    inv_gLretm = decimation.inv_g00(HL, TL, omega, eps, decimation_param, maxiter)
    t2 = time.time()
    return inv_gLretm, t2 - t1


# now not local to make it pickle-able
def calculate_mode_transmission(
    omega, HL, TL, KC, VLC, VCR, aL, eps, decimation_param, maxiter, tolerance
):
    # print(omega)
    time_int = {}
    for time_key in TIME_KEYS:
        time_int[time_key] = 0.0

    HR = HL.copy()
    TR = TL.copy()
    aR = aL
    HL_complex = HL.astype(complex)
    TL_complex = TL.astype(complex)
    HR_complex = HR.astype(complex)
    TR_complex = TR.astype(complex)
    en = omega * (omega + 1.0j * eps)
    # Build the four retarded GFs of leads extending left or right.
    # these are all equally sized problems
    inv_gLretm, treq = timed_decimation_inv_g00(
        HL_complex, TL_complex, omega, eps, decimation_param, maxiter
    )

    time_int["decimation"] += treq
    inv_gLretp, treq = timed_decimation_inv_g00(
        HL_complex,
        TL_complex.conj().T,
        omega,
        eps,
        decimation_param,
        maxiter,
    )
    time_int["decimation"] += treq
    inv_gRretm, treq = timed_decimation_inv_g00(
        HR_complex, TR_complex, omega, eps, decimation_param, maxiter
    )
    time_int["decimation"] += treq
    inv_gRretp, treq = timed_decimation_inv_g00(
        HR_complex,
        TR_complex.conj().T,
        omega,
        eps,
        decimation_param,
        maxiter,
    )
    time_int["decimation"] += treq
    t1 = time.time()
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
    Gret = la.pinv(en * np.eye(H_pr.shape[0], dtype=np.complex128) - H_pr, atol=1e-3)

    # Compute the total transmission.
    GammaL = 1.0j * TL.conj().T @ (la.solve(inv_gLretm, TL) - la.solve(inv_gLadvm, TL))
    GammaR = (
        1.0j
        * TR
        @ (la.solve(inv_gRretp, TR.conj().T) - la.solve(inv_gRadvp, TR.conj().T))
    )
    GLRret = Gret[: HL_pr.shape[0], -HR_pr.shape[1] :]
    GRLret = Gret[-HR_pr.shape[0] :, : HL_pr.shape[1]]

    trans = np.trace(GRLret @ (GRLret @ GammaL.conj().T).conj().T @ GammaR).real
    t2 = time.time()

    time_int["build_Gammas"] += t2 - t1

    # Build the Bloch matrices.
    FLretp = la.solve(inv_gLretp, TL.conj().T)
    FRretp = la.solve(inv_gRretp, TR.conj().T)
    FLadvp = la.solve(inv_gLadvp, TL.conj().T)
    FRadvp = la.solve(inv_gRadvp, TR.conj().T)
    inv_FLretm = la.solve(inv_gLretm, TL)
    inv_FRretm = la.solve(inv_gRretm, TR)
    inv_FLadvm = la.solve(inv_gLadvm, TL)
    inv_FRadvm = la.solve(inv_gRadvm, TR)

    t1 = time.time()
    time_int["build_Bloch"] += t1 - t2

    # Solve the corresponding eigenvalue equations for the leads.
    # Look for degenerate modes and orthonormalize them.
    def orthogonalize(values, vectors):
        modules = np.abs(values)
        # phases = np.angle(values)
        order = np.argsort(-modules)
        values = np.copy(values[order])
        vectors = np.copy(vectors[:, order])
        lo = 0
        hi = 1
        groups = []
        while True:
            if hi >= vectors.shape[1] or not np.isclose(
                np.angle(values[hi]), np.angle(values[hi - 1]), tolerance
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

    # Refine these matrices using the precomputed propagation masks.
    ALretp, ULretp = orthogonalize(*la.eig(FLretp))
    ARretp, URretp = orthogonalize(*la.eig(FRretp))
    ALadvp, ULadvp = orthogonalize(*la.eig(FLadvp))
    ARadvp, URadvp = orthogonalize(*la.eig(FRadvp))
    ALretm, ULretm = orthogonalize(*la.eig(inv_FLretm))
    ARretm, URretm = orthogonalize(*la.eig(inv_FRretm))
    ALadvm, ULadvm = orthogonalize(*la.eig(inv_FLadvm))
    ARadvm, URadvm = orthogonalize(*la.eig(inv_FRadvm))

    t2 = time.time()

    time_int["orthogonalize"] += t2 - t1

    # Find out which modes are propagating.
    mask_Lretp = np.isclose(np.abs(ALretp), 1.0, tolerance)
    mask_Rretp = np.isclose(np.abs(ARretp), 1.0, tolerance)
    mask_Ladvp = np.isclose(np.abs(ALadvp), 1.0, tolerance)
    mask_Radvp = np.isclose(np.abs(ARadvp), 1.0, tolerance)
    mask_Lretm = np.isclose(np.abs(ALretm), 1.0, tolerance)
    mask_Rretm = np.isclose(np.abs(ARretm), 1.0, tolerance)
    mask_Ladvm = np.isclose(np.abs(ALadvm), 1.0, tolerance)
    mask_Radvm = np.isclose(np.abs(ARadvm), 1.0, tolerance)

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
    # yapf: enable
    t1 = time.time()

    time_int["compute_group_velocities"] += t1 - t2

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

    t2 = time.time()
    time_int["refine_group_velocities"] += t2 - t1

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
    t1 = time.time()

    time_int["build_tilde"] += t1 - t2
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

    # Compute the number of right-propagating and left-propagating
    # channels, and perform a small sanity check.
    NLp = np.trace(ILretp).real
    NRp = np.trace(IRretp).real
    NLm = np.trace(ILretm).real
    NRm = np.trace(IRretm).real

    oNLp = np.trace(ILadvp).real
    oNRp = np.trace(IRadvp).real
    oNLm = np.trace(ILadvm).real
    oNRm = np.trace(IRadvm).real
    if not (
        np.isclose(NLp, oNLp)
        and np.isclose(NRp, oNRp)
        and np.isclose(NLm, oNLm)
        and np.isclose(NRm, oNRm)
    ):
        print("Warning: Inconsistent transmission for omega =", omega)
    t2 = time.time()

    time_int["build_channels"] += t2 - t1
    # NLp[iomega] = np.sum(mask_Lretp)
    # NRp[iomega] = np.sum(mask_Rretp)
    # NLm[iomega] = np.sum(mask_Ladvp)
    # NRm[iomega] = np.sum(mask_Radvp)

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
    ALretp = ALretp[mask_Lretp]
    ARretp = ARretp[mask_Rretp]
    ALadvp = ALadvp[mask_Ladvp]
    ARadvp = ARadvp[mask_Radvp]
    ALretm = ALretm[mask_Lretm]
    ARretm = ARretm[mask_Rretm]
    ALadvm = ALadvm[mask_Ladvm]
    ARadvm = ARadvm[mask_Radvm]

    probabilities = np.abs(tRL) ** 2
    # matrices.append(probabilities)

    # Compute the total transmission again.
    trans_check = np.diag(tRL.conj().T @ tRL).sum().real

    if not np.isclose(trans_check, trans, atol=1.0):
        print("Problem at omega={} rad/ps".format(omega))
    t1 = time.time()
    time_int["compute_transmission"] += t1 - t2

    # print(trans)
    # np.savetxt(
    #     f"inv_gLretm_parallel_{omega}.txt",
    #     np.concatenate(
    #         [
    #             GammaL.ravel(),

    #         ]
    #     ),
    # )
    # exit()
    return NLp, trans, time_int


def compute_transmission(
    HL,
    TL,
    KC,
    VLC,
    VCR,
    phonon,
    poscar,
    idx_indices,
    path_savefig,
    eps=1e-5,
    tolerance=1e-3,
    decimation_param=1e-8,
    maxiter=100000,
    n_procs=4,
    NPOINTS=21,
    noshow=False,
):
    t_init = time.time()
    mass_C = np.diag(
        np.power(
            ase.data.atomic_masses[phonon.primitive.numbers[idx_indices]], -1 / 2
        )
    )
    mass_L = np.diag(np.power(ase.data.atomic_masses[poscar.numbers], -1 / 2))
    mass_R = mass_L.copy()
    unit = 1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J

    HL = unit * np.einsum(
        "ijlm,jk->iklm", np.einsum("ij,jkln->ikln", mass_L, HL), mass_L
    )
    TL = unit * np.einsum(
        "ijlm,jk->iklm", np.einsum("ij,jkln->ikln", mass_L, TL), mass_L
    )
    KC = unit * np.einsum(
        "ijlm,jk->iklm", np.einsum("ij,jkln->ikln", mass_C, KC), mass_C
    )
    VLC = unit * np.einsum(
        "ijlm,jk->iklm", np.einsum("ij,jkln->ikln", mass_L, VLC), mass_C
    )
    VCR = unit * np.einsum(
        "ijlm,jk->iklm", np.einsum("ij,jkln->ikln", mass_C, VCR), mass_R
    )

    HL = HL.transpose((0, 2, 1, 3)).reshape((HL.shape[0] * 3, -1))
    TL = TL.transpose((0, 2, 1, 3)).reshape((TL.shape[0] * 3, -1))
    KC = KC.transpose((0, 2, 1, 3)).reshape((KC.shape[0] * 3, -1))
    VLC = VLC.transpose((0, 2, 1, 3)).reshape((VLC.shape[0] * 3, -1))
    VCR = VCR.transpose((0, 2, 1, 3)).reshape((VCR.shape[0] * 3, -1))

    aL = phonon.primitive.cell[2, 2]

    time_dict = {}
    for time_key in TIME_KEYS:
        time_dict[time_key] = 0.0

    # Plot the phonon spectra of both bulk leads.
    # qvec = np.linspace(0.0, 2.0 * np.pi, num=1001)
    qvec = np.linspace(-np.pi, np.pi, num=NPOINTS)
    t1 = time.time()
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    t2 = time.time()
    time_dict["q2omega"] = t2 - t1
    # Compute all the parameters of the interface
    inc_omega = np.linspace(1e-4, omegaL.max() * 1.01, num=NPOINTS, endpoint=True)

    trans_check = np.empty_like(inc_omega)
    # NRp = np.empty_like(inc_omega)
    # NLm = np.empty_like(inc_omega)
    # NRm = np.empty_like(inc_omega)

    if n_procs > 1:
        # turn off openmp parallelization
        # THIS DOES NOT WORK! NEEDS TO BE SET BEFORE EXECUTION!
        # there probably is a way to do it here, but I am unsure if it is worth figuring out
        os.environ["OMP_NUM_THREADS"] = "1"
    # for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
    #     calculate_mode_transmission(omega)
    partial_transmission = partial(
        calculate_mode_transmission,
        HL=HL,
        TL=TL,
        KC=KC,
        VLC=VLC,
        VCR=VCR,
        aL=aL,
        eps=eps,
        decimation_param=decimation_param,
        maxiter=maxiter,
        tolerance=tolerance,
    )

    with multiprocessing.Pool(n_procs) as pl:
        output = pl.map(partial_transmission, tqdm.tqdm(inc_omega, dynamic_ncols=True))

    trans = np.zeros_like(inc_omega)
    NLp = np.zeros_like(inc_omega)
    for iomega, (NLp_i, trans_i, time_dict_i) in enumerate(output):
        # print((NLp_i, trans_i))
        trans[iomega] = trans_i
        NLp[iomega] = NLp_i
        for key in time_dict.keys():
            time_dict[key] += time_dict_i[key]
    t_end = time.time()
    elapsed_time = t_end - t_init
    print("Elapsed time (real): {} s".format(elapsed_time))
    for key in time_dict.keys():
        print(
            f"\t{key:30s}: {time_dict[key]:8.3f} s {time_dict[key]/n_procs/elapsed_time*100:8.3f} %"
        )

    # np.savetxt(
    #     f"transmission_{NPOINTS}.txt",
    #     np.c_[inc_omega / 2 / np.pi, NLp, trans],
    # )

    fig1, axs1 = plt.subplots()

    plt.plot(inc_omega / 2 / np.pi, NLp, label=r"$Pure-N_{L+}$")
    plt.plot(inc_omega / 2 / np.pi, trans, label=r"$Defect-N_{L+}$")

    plt.xlabel("frequency / THz")
    plt.ylabel(r"$T(\omega)$")

    # plt.xlim(0, max(inc_omega))
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_savefig, dpi=600)
    if not noshow:
        plt.show()


def compute_transmission_from_files(
    data_directory, eps=1e-5, tolerance=1e-3, decimation_param=1e-8, maxiter=100000, n_procs=4
):
    path_directory = data_directory

    path_phonopy = os.path.join(path_directory, "phonopy_defect.yaml")
    path_LR_blocks = os.path.join(path_directory, "pure_fc.npz")
    path_scatter_blocks = os.path.join(path_directory, "scatter_fc.npz")
    path_defect_indices = os.path.join(path_directory, "defect_indices.npz")
    path_poscar = os.path.join(path_directory, "POSCAR")
    path_savefig = os.path.join(path_directory, "pure-defect")

    phonon = phonopy.load(phonopy_yaml=path_phonopy)
    poscar = ase.io.read(path_poscar)

    LR_blocks = np.load(path_LR_blocks)
    scatter_blocks = np.load(path_scatter_blocks)
    defect_indices = np.load(path_defect_indices)["defect_indices"]

    HL = LR_blocks["H00"]
    TL = LR_blocks["H01"]
    KC = scatter_blocks["Hc"]
    VLC = scatter_blocks["Vlc"]
    VCR = scatter_blocks["Vcr"]

    cells = max(defect_indices) + 1
    idx_scatter = np.where(defect_indices == int((cells - 1) / 2))[0]
    compute_transmission(
        HL,
        TL,
        KC,
        VLC,
        VCR,
        phonon,
        poscar,
        idx_scatter,
        path_savefig,
        eps,
        tolerance,
        decimation_param,
        maxiter,
        n_procs,
    )


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
        dest="decimation_param",
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
    parser.add_argument(
        "-n",
        "--n_procs",
        type=int,
        default=4,
        help="the number of multijobs",
    )
    # parser.add_argument("phonopy_file", help="phonopy yaml file")
    # parser.add_argument("pure_fc_file", help="force constant file")
    # parser.add_argument("scatter_fc_file", help="force constant file")
    # parser.add_argument("defect_indices", help="force constant file")
    parser.add_argument("data_directory", help="directory")
    args = parser.parse_args()

    compute_transmission_from_files(**vars(args))



