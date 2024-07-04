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
from utilities import divide_irreps, divide_over_irreps, get_adapted_matrix_multiq, get_adapted_matrix


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
        default=1e-6,
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
        default=1e-2,
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
        default=200000,
        help="maximum number of iterations in the decimation loop",
    )
    parser.add_argument("data_directory", help="directory")
    args = parser.parse_args()

    path_directory = args.data_directory
    path_phonopy_defect = os.path.join(path_directory, "phonopy_defect.yaml")
    path_phonopy_pure = os.path.join(path_directory, "phonopy_pure.yaml")
    path_fc_continum = os.path.join(path_directory, "FORCE_CONSTANTS_pure.continuum")

    path_LR_blocks = os.path.join(path_directory, "pure_fc.npz")
    path_scatter_blocks = os.path.join(path_directory, "scatter_fc.npz")
    path_defect_indices = os.path.join(path_directory, "defect_indices.npz")
    path_poscar = os.path.join(path_directory, "POSCAR")

    ######################### projector #######################
    # phonon = phonopy.load(phonopy_yaml=path_phonopy_pure, is_compact_fc=True)
    phonon = phonopy.load(phonopy_yaml=path_phonopy_pure, force_constants_filename=path_fc_continum, is_compact_fc=True)

    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)
    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    atom = cyclic._primitive
    atom_center = find_axis_center_of_nanotube(atom)
    num_atom = len(atom_center)
    ################ family 4 ##################
    family = 4
    num_irreps = 12
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    sym  = []
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    # pg1 = obj.get_generators()
    # sym.append(pg1[0])
    rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    sym.append(tran.affine_matrix)
    sym.append(rots.affine_matrix)
    sym.append(mirror.affine_matrix)
    ################### family 2 #############
    # family = 2
    # num_irreps = 6
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.get_rotational_symmetry_number()
    # sym = []
    # # pg1 = obj.get_generators()  # change the order to satisfy the character table
    # # sym.append(pg1[1])
    # rots = SymmOp.from_rotation_and_translation(S2n(nrot), [0, 0, 0])
    # sym.append(rots.affine_matrix)
    #########################################
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-6)
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

    # aL = phonon.primitive.cell[2, 2]
    aL = cyclic._pure_trans
    aR = aL

    k_start = -np.pi+0.1
    # k_start = 0
    k_end = np.pi-0.1

    # Plot the phonon spectra of both bulk leads.
    # qvec = np.linspace(0.0, 2.0 * np.pi, num=1001)
    qvec = np.linspace(k_start, k_end, num=NPOINTS*10)
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

    # k_res1, k_res2 = [], []
    # for iomega, omega in enumerate(tqdm.tqdm(inc_omega, dynamic_ncols=True)):
    # omega = 7.656578907447281
    omega = 17.102302541756387
    print("-----------")
    print("omega = ", omega)
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

    ########## Move the calculation of adapted matrix out of "orthogonalize" function   ############
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
    # k_test = np.linspace(0, (np.pi-0.1)/aL, 10, endpoint=True)
    # adapteds_test, dimensions_test = get_adapted_matrix_multiq(k_test, nrot, order_ops, family, aL, num_atoms, matrices)

    irreps = []
    if mask.sum() != 0:  # not all False
        # k_w = np.abs(np.angle(values[mask])) / aL
        k_w = np.arccos(values[mask].real) / aL
        k_adapteds = np.unique(k_w)

        adapteds, dimensions = [], []
        for qp in k_adapteds:
            DictParams = {"qpoints":qp,  "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13
            adapted, dimension = get_adapted_matrix(DictParams, num_atom, matrices)
            adapteds.append(adapted)
            dimensions.append(dimension)

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

        # mask = np.isclose(np.abs(values), 1.0, args.rtol, args.atol)
        index = np.arange(len(mask))
        group_masks = []
        for g in groups:
            lo, hi = g
            group_masks.append(mask & (index >= lo) & (index < hi))

        irreps = []
        for i_m, m in enumerate(group_masks):
            degeneracy = m.sum()
            if degeneracy == 0:
                continue

            # k_w_group = np.abs(np.angle(values[m][0])) / aL
            k_w_group = np.arccos(values[m].real)[0] / aL

            try:
                k_w_group_indice = np.where(np.isclose(k_w_group, k_adapteds))[0].item()
            except:
                set_trace()
                logging.ERROR("No correspond k index")
            basis, dims = adapteds[k_w_group_indice], dimensions[k_w_group_indice]
            group_vectors = vectors[:, m]

            try:
                adapted_vecs = divide_over_irreps(group_vectors, basis, dims)
            except Exception as e:
                print(e)
                res = divide_irreps(group_vectors.T,basis,dims)
                print(res)
                set_trace()
                continue

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
