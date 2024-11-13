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
from sympy.codegen.ast import continue_

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
from utilities import counting_y_from_xy, get_adapted_matrix, divide_irreps, divide_over_irreps, get_adapted_matrix_multiq
import matplotlib.colors as mcolors


matplotlib.rcParams["font.size"] = 16.0
NPOINTS = 11

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
    ################### family 2 ###############
    # family = 2
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.get_rotational_symmetry_number()
    # num_irreps = nrot * 2
    # sym = []
    # # pg1 = obj.get_generators()  # change the order to satisfy the character table
    # # sym.append(pg1[1])
    # rots = SymmOp.from_rotation_and_translation(S2n(nrot), [0, 0, 0])
    # sym.append(rots.affine_matrix)
    ############################################
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

    trans = np.zeros_like(inc_omega)
    trans_check = np.zeros_like(inc_omega)

    NLp = np.zeros_like(inc_omega)
    NRp = np.zeros_like(inc_omega)
    NLm = np.zeros_like(inc_omega)
    NRm = np.zeros_like(inc_omega)

    t1 = time.time()
    print("The time of load data: %s" % (t1 - t0))

    HL_complex = HL.astype(complex)
    TL_complex = TL.astype(complex)
    HR_complex = HR.astype(complex)
    TR_complex = TR.astype(complex)
    matrices_prob, Irreps = [], []
    NLp_irreps = np.zeros((num_irreps, NPOINTS))  # the number of im
    # for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
    iomega, omega = 0, 21.18771565358862

    print("---------------------")
    print("omega=", omega)

    t0 = time.time()
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

    print(trans[iomega])
    t1 = time.time()
    print("The time of G matrix: %s" % (t1-t0))
    def orthogonalize(values, vectors, DictParams, k_adapteds, adapteds, dimensions, output=True, inv_index=False):
        mask = np.isclose(np.abs(values), 1.0, args.rtol, args.mtol)

        # order_val = np.lexsort((np.arccos(values.real), 1 * (~mask)))
        order_val = np.lexsort((np.angle(values), 1 * (~mask)))
        values = values[order_val]
        vectors = vectors[:, order_val]

        lo = 0
        hi = 1
        groups = []
        while True:
            if mask.sum() == 0:
                break

            if hi >= mask.sum() or not np.isclose(
            np.angle(values[hi]) / aL, np.angle(values[hi - 1]) / aL, rtol=args.rtol, atol=args.atol,
            # np.arccos(values[hi].real) / aL, np.arccos(values[hi - 1].real) / aL, rtol=args.rtol, atol=args.atol,
            ):
                groups.append((lo, hi))
                lo = hi
            if hi >= mask.sum():
                break
            hi += 1
        for g in groups:
            lo, hi = g
            if hi > lo + 1:
                values[lo:hi] = values[lo:hi].mean()
                vectors[:, lo:hi] = la.orth(vectors[:, lo:hi])

        irreps = []
        for g in groups:
            lo, hi = g

            # k_w_group = np.arccos(values[lo].real) / aL
            k_w_group = np.angle(values[lo]) / aL
            DictParams["qpoints"] = k_w_group

            if k_w_group in k_adapteds:
                itp = k_adapteds.index(k_w_group)
                basis, dims = adapteds[itp], dimensions[itp]
            else:
                tmp1 = phonon.primitive.positions[:, 2].reshape(-1, 1)
                factor_pos = tmp1 - tmp1.T
                factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1).astype(np.complex128)
                if inv_index == True:
                    factor_pos = factor_pos.T
                matrices = get_matrices_withPhase(atom_center, ops_car_sym, k_w_group)
                matrices = matrices * np.exp(1j * k_w_group * factor_pos)

                basis, dims = get_adapted_matrix(DictParams, num_atoms, matrices)
                k_adapteds.append(k_w_group)
                adapteds.append(basis)
                dimensions.append(dims)

            group_vectors = vectors[:, lo:hi]
            D2 = phonon.get_dynamical_matrix_at_q([0, 0, k_w_group * aL / 2 / np.pi])

            D3 = TL1.conj().transpose() * np.exp(-1j * k_w_group * aL) + HL1 + TL1 * np.exp(1j * k_w_group * aL)
            eigvals3, eigvecs3 = np.linalg.eigh(D3)
            eigvals3 = eigvals3.real
            frequency3 = np.sqrt(abs(eigvals3)) * np.sign(eigvals3) * VaspToTHz * 2 * np.pi

            D_sym = basis.conj().T @ D3 @ basis
            start = 0
            adapted_vecs, adapteds_values, adapteds_freq = [], [], []
            for dim in dims:
                end = start + dim
                eigvals, eigvecs = np.linalg.eigh(D_sym[start:end, start:end])
                adapteds_values.append(eigvals)
                adapted_vecs.append(basis[:, start:end] @ eigvecs)
                eigvals = eigvals.real
                adapteds_freq.append(np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz * 2 * np.pi)
                start = np.copy(end)
            adapteds_values = np.concatenate(adapteds_values)
            adapteds_freq = np.concatenate(adapteds_freq)
            adapted_vecs = np.concatenate(adapted_vecs, axis=1)

            eigvals2, eigvecs2 = np.linalg.eigh(D2)
            eigvals2 = eigvals2.real
            frequency2 = np.sqrt(abs(eigvals2)) * np.sign(eigvals2) * VaspToTHz * 2 * np.pi


            idx2 = np.argsort(np.abs(frequency2 - omega))[:(hi - lo)]
            idx3 = np.argsort(np.abs(frequency3 - omega))[:(hi - lo)]
            idx4 = np.argsort(np.abs(adapteds_freq - omega))[:(hi - lo)]

            group_vectors_D2 = eigvecs2[:, idx2]
            group_vectors_D3 = eigvecs3[:, idx3]
            group_vectors_Dsym = adapted_vecs[:, idx4]

            freq1 = omega
            freq2 = frequency2[idx2]
            freq3 = frequency3[idx3]
            freq4 = adapteds_freq[idx4]

            try:
                adapted_vecs = divide_over_irreps(group_vectors, basis, dims, rcond=rcond)
            except Exception as e:
                print("k_w_group:", k_w_group)
                print(e)

                res1 = divide_irreps(group_vectors.T, basis, dims)      # the projection of Bloch matrix
                res2 = divide_irreps(group_vectors_D2.T, basis, dims)        # the projection of phase convention D
                res3 = divide_irreps(group_vectors_D3.T, basis, dims)           # the projection of no phase convention D
                res4 = divide_irreps(group_vectors_Dsym.T, basis, dims)        # the projection of adapted no phase D

                print("freq1:", freq1, "res1:", res1)
                print("freq3:", freq3, "res3:", res1)
                # print("res3": res3)
                set_trace()



            # def clip_fc(comx):
            #     return np.sign(comx.real) * np.clip(np.abs(comx.real), a_min=1e-5, a_max=None) + 1j * np.sign(
            #         comx.imag) * np.clip(np.abs(comx.imag), a_min=1e-5, a_max=None)
            # group_vectors = clip_fc(group_vectors)
            # group_vectors_D1 = clip_fc(group_vectors_D1)
            # group_vectors_Dsym = clip_fc(group_vectors_Dsym)
            # group_vectors_D3 = clip_fc(group_vectors_D3)


            # phase12 = np.angle(np.exp(1j * (np.angle(group_vectors) - np.angle(group_vectors_D1))))
            # phase14 = np.angle(np.exp(1j * (np.angle(group_vectors) - np.angle(group_vectors_D3))))
            # phase23 = np.angle(np.exp(1j * (np.angle(group_vectors_Dsym) - np.angle(group_vectors_D3))))
            # phase24 = np.angle(np.exp(1j * (np.angle(group_vectors_D1) - np.angle(group_vectors_D3))))
            # phase_D13 = np.angle(np.exp(1j * (np.angle(D1) - np.angle(D3))))

        return values, vectors, mask, irreps, k_adapteds, adapteds, dimensions

        # # Solve the corresponding eigenvalue equations for the leads.
        # # Look for degenerate modes and orthonormalize them.
    DictParams = {"nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13

    k_adapteds, adapteds, dimensions = [], [], []
    ALadvm, ULadvm, mask_Ladvm, tmp_irreps, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(inv_FLadvm), DictParams, k_adapteds, adapteds, dimensions,inv_index=True)
    k_adapteds, adapteds, dimensions = [], [], []
    ALretp, ULretp, mask_Lretp, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(FLretp), DictParams, k_adapteds, adapteds, dimensions, inv_index=False)
