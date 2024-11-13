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
NPOINTS = 13

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
    phonon = phonopy.load(phonopy_yaml=path_phonopy_pure)
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

    # frequencies_raw1, frequencies_raw2 = [], []
    # for ii, qp in enumerate(qpoints_1dim):   # loop q points
    #
    #     qz = qpoints[ii]
    #     D1 = phonon.get_dynamical_matrix_at_q(qz)
    #     eigvals1, eigvecs1 = np.linalg.eigh(D1)
    #     eigvals1 = eigvals1.real
    #     frequencies_raw1.append(np.sqrt(abs(eigvals1)) * np.sign(eigvals1) * VaspToTHz)
    #
    #     # D2 = TL.conj().transpose() * np.exp(-1j*qp*aL) + HL + TL * np.exp(1j*qp*aL)
    #     # eigvals2, eigvecs2 = np.linalg.eigh(D2)
    #     # eigvals2 = eigvals2.real
    #     # frequencies_raw2.append(np.sqrt(abs(eigvals2)) * np.sign(eigvals2) * VaspToTHz)
    # frequencies_raw1 = np.array(frequencies_raw1).T * 2 * np.pi
    # # frequencies_raw2 = np.array(frequencies_raw2).T * 2 * np.pi
    #
    # fig, ax = plt.subplots()
    # for i, f_raw in enumerate(frequencies_raw1):
    #     if i == 0:
    #         ax.plot(qpoints_1dim, f_raw, color="grey", label="raw")
    #         ax.plot(qpoints_1dim, omegaL.T[i], color="red", label="decimation")
    #     else:
    #         ax.plot(qpoints_1dim, f_raw, color="grey")
    #         ax.plot(qpoints_1dim, omegaL.T[i], color="red")
    #
    # plt.savefig(path_save_phonon, dpi=600)
    # plt.show()

    trans = np.zeros_like(inc_omega)
    trans_check = np.zeros_like(inc_omega)

    NLp = np.zeros_like(inc_omega)
    NRp = np.zeros_like(inc_omega)
    NLm = np.zeros_like(inc_omega)
    NRm = np.zeros_like(inc_omega)

    t1 = time.time()
    print("Load data: %s" % (t1 - t0))

    HL_complex = HL.astype(complex)
    TL_complex = TL.astype(complex)
    HR_complex = HR.astype(complex)
    TR_complex = TR.astype(complex)
    matrices_prob, Irreps = [], []
    NLp_irreps = np.zeros((num_irreps, NPOINTS))  # the number of im
    for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
        # iomega, omega = 0, 15.846153846153847

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
        def orthogonalize(values, vectors, DictParams, k_adapteds, adapteds, dimensions, output=True):
            mask = np.isclose(np.abs(values), 1.0, args.rtol, args.mtol)

            # order_val = np.lexsort((np.arccos(values.real), 1 * (~mask)))
            order_val = np.lexsort((np.angle(values), 1 * (~mask)))
            values = values[order_val]
            vectors = vectors[:, order_val]

            lo = 0
            hi = 1
            groups = []
            while True:
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
                    factor_pos = np.repeat(np.repeat(factor_pos, 3, axis=0), 3, axis=1)
                    matrices = get_matrices_withPhase(atom_center, ops_car_sym, k_w_group)
                    matrices = matrices * np.exp(1j * k_w_group * factor_pos)

                    # matrices = get_matrices_withPhase(atom_center, ops_car_sym, k_w_group)

                    basis, dims = get_adapted_matrix(DictParams, num_atoms, matrices)
                    k_adapteds.append(k_w_group)
                    adapteds.append(basis)
                    dimensions.append(dims)

                group_vectors = vectors[:, lo:hi]
                try:
                    adapted_vecs = divide_over_irreps(group_vectors, basis, dims, rcond=rcond)
                    tmp_vec = []
                    for i_ir, v in enumerate(adapted_vecs):
                        if v.shape[1] > 0:
                            tmp_vec.append(v)
                            irreps.extend(np.repeat(i_ir, v.shape[1]))
                            # print(f"\t- {v.shape[1]} modes in irrep #{i_ir+1}")
                    tmp_vec = np.concatenate(tmp_vec, axis=1)
                    vectors[:, lo:hi] = tmp_vec.copy()

                except Exception as e:
                    D1 = phonon.get_dynamical_matrix_at_q([0, 0, k_w_group * aL / 2 / np.pi])
                    eigvals1, eigvecs1 = np.linalg.eigh(D1)
                    eigvals1 = eigvals1.real
                    frequency1 = np.sqrt(abs(eigvals1)) * np.sign(eigvals1) * VaspToTHz * 2 * np.pi


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


                    idx1 = np.argsort(np.abs(frequency1 - omega))[:(hi-lo)]
                    idx2 = np.argsort(np.abs(adapteds_freq - omega))[:(hi-lo)]
                    idx3 = np.argsort(np.abs(frequency3 - omega))[:(hi-lo)]

                    group_vectors_D1 = eigvecs1[:, idx1]
                    group_vectors_Dsym = adapted_vecs[:, idx2]
                    group_vectors_D3 = eigvecs3[:, idx3]

                    def clip_fc(comx):
                        return np.sign(comx.real)*np.clip(np.abs(comx.real), a_min=1e-5, a_max=None) + 1j*np.sign(comx.imag)*np.clip(np.abs(comx.imag), a_min=1e-5, a_max=None)

                    group_vectors = clip_fc(group_vectors)
                    group_vectors_D1 = clip_fc(group_vectors_D1)
                    group_vectors_Dsym = clip_fc(group_vectors_Dsym)
                    group_vectors_D3 = clip_fc(group_vectors_D3)

                    res0 = divide_irreps(group_vectors.T, basis, dims)
                    res1 = divide_irreps(group_vectors_D1.T, basis, dims)
                    res2 = divide_irreps(group_vectors_Dsym.T, basis, dims)
                    res3 = divide_irreps(group_vectors_D3.T, basis, dims)



                    if output ==True:
                        print("k_w_group:", k_w_group)
                        print("omega:", omega)
                        print(e)
                        print("res2=", res2)
                        print("res1=", res1)

                    vectors[:, lo:hi] = group_vectors_Dsym.copy()
                    values[lo:hi] = adapteds_values[idx1].copy()

                    # if hi-lo==1:
                    #     irreps.append(np.where(np.isclose(res1,1))[0].item())
                    # else:
                    #     tmp_irreps = [tmp[1] for tmp in np.where(np.isclose(res1, 1, atol=args.atol))]
                    #     irreps.extend(tmp_irreps)

            return values, vectors, mask, irreps, k_adapteds, adapteds, dimensions

        # # Solve the corresponding eigenvalue equations for the leads.
        # # Look for degenerate modes and orthonormalize them.
        DictParams = {"nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13

        k_adapteds, adapteds, dimensions = [], [], []
        ALadvm, ULadvm, mask_Ladvm, tmp_irreps, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(inv_FLadvm), DictParams, k_adapteds, adapteds, dimensions)

        ALretm, ULretm, mask_Lretm, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(inv_FLretm), DictParams, k_adapteds, adapteds, dimensions,output=False)
        ARretm, URretm, mask_Rretm, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(inv_FRretm), DictParams, k_adapteds, adapteds, dimensions,output=False)
        ARadvm, URadvm, mask_Radvm, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(inv_FRadvm), DictParams, k_adapteds, adapteds, dimensions,output=False)
        ALretp, ULretp, mask_Lretp, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(FLretp), DictParams, k_adapteds, adapteds, dimensions,output=False)
        ARretp, URretp, mask_Rretp, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(FRretp), DictParams, k_adapteds, adapteds, dimensions,output=False)
        ALadvp, ULadvp, mask_Ladvp, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(FLadvp), DictParams, k_adapteds, adapteds, dimensions,output=False)
        ARadvp, URadvp, mask_Radvp, _, k_adapteds, adapteds, dimensions = orthogonalize(*la.eig(FRadvp), DictParams, k_adapteds, adapteds, dimensions,output=False)

        t2 = time.time()
        print("The orthogonalization process: %s" % (t2-t1))

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

        # Compute the number of right-propagating and left-propagating
        # channels, and perform a small sanity check.
        NLp[iomega] = np.trace(ILretp).real
        NRp[iomega] = np.trace(IRretp).real
        NLm[iomega] = np.trace(ILretm).real
        NRm[iomega] = np.trace(IRretm).real


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

        # matrices_prob.append(np.diag(trans_modes))
        matrices_prob.append(np.abs(tRL) ** 2)
        for im, tras in enumerate(trans_modes):
            if tras > 100:
                set_trace()

            try:
                NLp_irreps[tmp_irreps[im], iomega] += tras
            except:

                set_trace()
                print("NLp_irreps.shape=", NLp_irreps.shape)
                print("iomega=", iomega)
                print("tmp_irreps.shape", tmp_irreps.shape)
                print("im=", im)
        t3 = time.time()
        print("post-process: %s" % (t3-t2))
    NLp_sum = NLp_irreps.sum(axis=0)

    t0 = time.time()
    fig, axs = plt.subplots(figsize=(12, 8))

    # colors = [value for key, value in mcolors.TABLEAU_COLORS.items()]
    # colors = [value for key, value in mcolors.CSS4_COLORS.items()]
    colors = [value for key, value in mcolors.XKCD_COLORS.items()]
    # color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan']
    labels = ["|m|=0", "|m|=1", "|m|=2", "|m|=3", "|m|=4", "|m|=5", "|m|=6", "|m|=7", "|m|=8", "|m|=9", "|m|=10", "|m|=11"]
    linestyle_tuple = ['solid',
        ('long dash with offset', (5, (10, 3))),
        ('loosely dashed', (0, (5, 10))),]

    # NLp_irreps = np.array([NLp_irreps[2], NLp_irreps[1] + NLp_irreps[3], NLp_irreps[0] + NLp_irreps[4], NLp_irreps[5]])

    set_trace()
    for ii, freq in enumerate(NLp_irreps):
        plt.plot(np.array(inc_omega), freq, label=labels[ii], color=colors[ii])

    plt.plot(inc_omega, NLp, label=r"$Pure-N_{L+}$", color="grey")
    plt.plot(np.array(inc_omega), trans, label="Caroli", color=colors[ii+1], linestyle=linestyle_tuple[0])
    plt.plot(np.array(inc_omega), trans_check, label="modes_sum", color=colors[ii+2], linestyle=linestyle_tuple[1][1])
    plt.plot(np.array(inc_omega), NLp_sum, label="Irreps_sum", color=colors[ii+3], linestyle=linestyle_tuple[2][1])

    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
    plt.ylabel(r"$T(\omega)$")
    plt.tick_params(labelsize=14)
    plt.savefig(os.path.join(path_directory, "transmission_sym_adapted_defect_21.png"), dpi=600)
    np.savez(path_savedata, inc_omega=inc_omega, NLp_irreps=NLp_irreps, NLp=NLp, NRp=NRp, NLm=NLm, NRm=NRm, trans=trans, trans_check=trans_check, NLp_sum=NLp_sum)

    t1 = time.time()
    print("plot and save the figure: %s" % (t1 - t0))
    plt.show()
