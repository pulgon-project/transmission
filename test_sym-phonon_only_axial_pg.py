import logging
import os.path
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import pretty_errors
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.utils import (
    get_matrices,
    find_axis_center_of_nanotube,
    Cn,
    S2n,
    brute_force_generate_group_subsquent,
)
from pymatgen.core.operations import SymmOp
from tqdm import tqdm
from utilities import counting_y_from_xy, get_adapted_matrix, commuting
import decimation
from spglib import get_symmetry_dataset
import ase
import scipy
from sympy.physics.quantum import TensorProduct


def ir_table():
    GM11 = [1,1,1,1,1,1,1,1,1,1,1,1]
    GM12 = [1,1,-1,1,-1,1,-1,1,-1,1,-1,-1]

    GM21 = [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1]
    GM22 = [1,-1,1,1,-1,-1,1,1,-1,-1,1,-1]

    GM31 = [1,np.exp(-1j*2*np.pi/3),1,np.exp(1j*2*np.pi/3),np.exp(1j*2*np.pi/3),1,np.exp(-1j*2*np.pi/3),np.exp(-1j*2*np.pi/3),1,np.exp(1j*2*np.pi/3),np.exp(1j*2*np.pi/3),np.exp(-1j*2*np.pi/3)]
    GM32 = [1,np.exp(-1j*2*np.pi/3),-1,np.exp(1j*2*np.pi/3),np.exp(-1j*np.pi/3),1,np.exp(1j*np.pi/3),np.exp(-1j*2*np.pi/3),-1,np.exp(1j*2*np.pi/3),np.exp(-1j*np.pi/3),np.exp(1j*np.pi/3)]

    GM41 = [1,np.exp(1j*np.pi/3),-1,np.exp(1j*2*np.pi/3),np.exp(1j*2*np.pi/3),-1,np.exp(1j*np.pi/3),np.exp(-1j*2*np.pi/3),1,np.exp(-1j*np.pi/3),np.exp(-1j*np.pi/3),np.exp(-1j*2*np.pi/3)]
    GM42 = [1,np.exp(1j*np.pi/3),1,np.exp(1j*2*np.pi/3),np.exp(-1j*np.pi/3),-1,np.exp(-1j*2*np.pi/3),np.exp(-1j*2*np.pi/3),-1,np.exp(-1j*np.pi/3),np.exp(1j*2*np.pi/3),np.exp(1j*np.pi/3)]

    GM51 = [1,np.exp(1j*2*np.pi/3),1,np.exp(-1j*2*np.pi/3),np.exp(-1j*2*np.pi/3),1,np.exp(1j*2*np.pi/3),np.exp(1j*2*np.pi/3),1,np.exp(-1j*2*np.pi/3),np.exp(-1j*2*np.pi/3),np.exp(1j*2*np.pi/3)]
    GM52 = [1,np.exp(1j*2*np.pi/3),-1,np.exp(-1j*2*np.pi/3),np.exp(1j*np.pi/3),1,np.exp(-1j*np.pi/3),np.exp(1j*2*np.pi/3),-1,np.exp(-1j*2*np.pi/3),np.exp(1j*np.pi/3),np.exp(-1j*np.pi/3)]

    GM61 = [1,np.exp(-1j*np.pi/3),-1,np.exp(-1j*2*np.pi/3),np.exp(-1j*2*np.pi/3),-1,np.exp(-1j*np.pi/3),np.exp(1j*2*np.pi/3),1,np.exp(1j*np.pi/3),np.exp(1j*np.pi/3),np.exp(1j*2*np.pi/3)]
    GM62 = [1,np.exp(-1j*np.pi/3),1,np.exp(-1j*2*np.pi/3),np.exp(1j*np.pi/3),-1,np.exp(1j*2*np.pi/3),np.exp(1j*2*np.pi/3),-1,np.exp(1j*np.pi/3),np.exp(-1j*2*np.pi/3),np.exp(-1j*np.pi/3)]

    res = np.array([GM11, GM12, GM21, GM22, GM31, GM32, GM41, GM42, GM51, GM52, GM61, GM62])
    return res



def get_std_projector_of_molecular(matrices_apg, atom):
    GM = ir_table()

    basis, dimensions = [], []
    for i_Dmu, Dmu_rot in enumerate(GM):
        # the degeneracy of IR
        d_mu = 1
        Dmu_rot = np.array(Dmu_rot).astype(np.complex128)

        ###### generate the projector for axial point group ########
        num_modes = 0
        ndof = 3 * len(atom)
        projector = np.zeros((ndof, ndof), dtype=np.complex128)
        for ii in range(len(Dmu_rot)):
            if d_mu == 1:
                num_modes += Dmu_rot[ii].conj() * matrices_apg[ii].trace()
                projector += Dmu_rot[ii].conj() * matrices_apg[ii]
            else:
                num_modes += Dmu_rot[ii].conj().trace() * matrices_apg[ii].trace()
                projector += Dmu_rot[ii].conj().trace() * matrices_apg[ii]


        num_modes = int(num_modes.real * d_mu / len(Dmu_rot))
        projector = d_mu * projector / (len(Dmu_rot))

        if num_modes ==0:
            dimensions.append(num_modes)
            continue
        u, s, vh = scipy.linalg.svd(projector)
        error = 1 - np.abs(s[num_modes - 1] - s[num_modes]) / np.abs(s[num_modes - 1])

        if error > 0.05:
            print("the error is %s" % error)
            # set_trace()
        # print("m=%s" % tmp_m1, "error=%s" % error)
        basis.append(u[:,:num_modes])
        dimensions.append(num_modes)
    adapted = np.concatenate(basis, axis=1)
    if adapted.shape[0] != adapted.shape[1]:
        print("the number of eigenvector is %d" % adapted.shape[1], "%d" % adapted.shape[0] + "is required")
        set_trace()
    return adapted, dimensions


def get_modified_projector_of_molecular(matrices_apg, atom):
    GM = ir_table()   # the order of Gm correspond to the matrices_apg

    basis, dimensions = [], []
    for i_Dmu, Dmu_rot in enumerate(GM):
        # the degeneracy of IR
        d_mu = 1
        ###### generate the projector for point group ########
        num_modes = 0
        for ii in range(len(Dmu_rot)):
            if ii == 0:
                projector = TensorProduct(np.array(Dmu_rot[ii].conj()), matrices_apg[ii])
            else:
                projector += TensorProduct( np.array(Dmu_rot[ii].conj()), matrices_apg[ii])
            if d_mu==1:
                num_modes += Dmu_rot[ii].conj() * matrices_apg[ii].trace()
            else:
                num_modes += Dmu_rot[ii].conj().trace() * matrices_apg[ii].trace()

        num_modes = int(num_modes.real / len(Dmu_rot))   # the number of modes for each IR
        projector = projector / (len(Dmu_rot))

        u, s, vh = scipy.linalg.svd(projector)
        error = 1 - np.abs(s[num_modes - 1] - s[num_modes]) / np.abs(s[num_modes - 1])
        if error > 0.05:
            print("error: ", error)
            # set_trace()

        if num_modes ==0:
            dimensions.append(num_modes)
            continue

        if Dmu_rot[ii].ndim == 0:
            basis.append(u[:, :num_modes])
            dimensions.append(num_modes)
        else:
            tmp_basis = u[:, :num_modes]

            tmp_basis1 = np.array(np.array_split(tmp_basis, len(Dmu_rot[0]), axis=0))
            basis_Dmu = np.abs(scipy.linalg.orth(Dmu_rot[ii]))
            # basis_Dmu = np.array([[0, 1], [1, 0]])

            ###### partial scalar product ######
            for ii in range(d_mu):
                basis_block1 = np.einsum("ij,jlm->ilm", basis_Dmu[:, ii][np.newaxis], tmp_basis1)[0]

                basis_block1 = basis_block1 / np.linalg.norm(basis_block1)

                # set_trace()
                basis.append(basis_block1)
            dimensions.append(d_mu* tmp_basis1.shape[2])

            # basis_block1 = np.einsum("ij,jlm->ilm", basis_Dmu[:, 0][np.newaxis], tmp_basis1)[0]
            # basis.append(basis_block1)
            # dimensions.append(basis_block1.shape[1])

    adapted = np.concatenate(basis, axis=1)
    if adapted.shape[0] != adapted.shape[1]:
        print("the number of eigenvector is %d" % adapted.shape[1], ", but %d" % adapted.shape[0] + " is required")
    return adapted, dimensions



def main():
    # path_0 = "datas/WS2/6-6-u1-3-defect-1"
    # path_0 = "datas/WS2/6-6-u2-5-defect-1"
    # path_0 = "datas/carbon_nanotube/4x1-u1-3-defect-C-1"
    path_0 = "datas/carbon_nanotube/3x3-6x6-u1-3-defect-C-1"
    path_0 = "datas/carbon_nanotube/5x0-10x0-u1-3-defect-C-1"
    path_yaml = os.path.join(path_0, "phonopy_pure.yaml")
    path_fc_continum = os.path.join(path_0, "FORCE_CONSTANTS_pure.continuum")
    path_save_phonon = os.path.join(path_0, "phonon_defect_sym_adapted")
    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)
    # phonon = phonopy.load(phonopy_yaml=path_yaml, is_compact_fc=True)
    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)

    # poscar_ase = read_vasp(os.path.join(path_0, "POSCAR"))

    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    aL = poscar_ase.cell[2,2]
    atom = cyclic._atom
    atom_center = find_axis_center_of_nanotube(atom)

    NQS = 51
    k_start = -np.pi + 0.1
    k_end = np.pi - 0.1

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    qpoints = qpoints[0]
    qpoints_1dim = qpoints[:,2] * 2 * np.pi
    qpoints_1dim = qpoints_1dim / aL

    #################### harmonic matrix #################
    path_LR_blocks = os.path.join(path_0, "pure_fc.npz")
    LR_blocks = np.load(path_LR_blocks)
    HL = LR_blocks["H00"]
    TL = LR_blocks["H01"]
    mass_L = np.diag(np.power(ase.data.atomic_masses[poscar_ase.numbers], -1/2))
    unit = 1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J
    # unit = 1
    HL1 = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, HL), mass_L)
    HL = np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, HL), mass_L)
    TL1 = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, TL), mass_L)
    TL = np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, TL), mass_L)
    HL1 = HL1.transpose((0, 2, 1, 3)).reshape((HL1.shape[0] * 3, -1))
    HL = HL.transpose((0, 2, 1, 3)).reshape((HL.shape[0] * 3, -1))
    TL1 = TL1.transpose((0, 2, 1, 3)).reshape((TL1.shape[0] * 3, -1))
    TL = TL.transpose((0, 2, 1, 3)).reshape((TL.shape[0] * 3, -1))
    # qvec = np.linspace(k_start, k_end, num=NQS)
    # omegaL, vgL = decimation.q2omega(HL1, TL1, qvec)
    ########################################################
    ################ family 4 ##################
    family = 4
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    sym  = []
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    mirror = SymmOp.reflection([0,0,1], [0,0,0])
    # mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    # sym.append(tran.affine_matrix)
    sym.append(rots.affine_matrix)
    sym.append(mirror.affine_matrix)

    set_trace()
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-5)
    ops_car_sym = []
    for op in ops:
        tmp_sym1 = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym1)
    matrices = get_matrices(poscar_ase, ops_car_sym)


    adapted, dimensions = get_std_projector_of_molecular(matrices, poscar_ase)
    # adapted, dimensions = get_modified_projector_of_molecular(matrices, atom_center)

    frequencies, distances, bands = [], [], []
    num_atom = len(poscar_ase.numbers)
    for ii, qp in enumerate(tqdm(qpoints_1dim)):   # loop q points
        qz = qpoints[ii]

        D = phonon.get_dynamical_matrix_at_q(qz)
        # D = TL.conj().transpose() * np.exp(-1j*qp*aL) + HL + TL * np.exp(1j*qp*aL)
        D = adapted.conj().T @ D @ adapted

        start = 0
        tmp_band = []
        for ir in range(len(dimensions)):
            end = start + dimensions[ir]
            block = D[start:end, start:end]
            eig, eigvecs = np.linalg.eigh(block)
            e = (
                    np.sqrt(np.abs(eig))
                    * np.sign(eig)
                    * VaspToTHz
            ).tolist()
            tmp_band.append(e)
            start = end
        bands.append(np.concatenate(tmp_band))
        distances.append(qp)
    frequencies = np.array(bands).swapaxes(0, 1) * 2 * np.pi

    fig, ax = plt.subplots()
    frequencies_raw = []
    for ii, q in enumerate(qpoints):
        # D = phonon.get_dynamical_matrix_at_q(q)
        qp = qpoints_1dim[ii]
        D = TL.conj().transpose() * np.exp(-1j*qp*aL) + HL + TL * np.exp(1j*qp*aL)

        eigvals, eigvecs = np.linalg.eigh(D)
        eigvals = eigvals.real
        frequencies_raw.append(np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz)
    frequencies_raw = np.array(frequencies_raw).T * 2 * np.pi
    ### raw phonon
    for i, f_raw in enumerate(frequencies_raw):
        if i == 0:
            ax.plot(distances, f_raw, color="grey", label="raw")
        else:
            ax.plot(distances, f_raw, color="grey")

    #### plot adapted phonon
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'yellow', 'pink', 'olive', 'sage', 'slategray', 'darkkhaki', 'yellowgreen']
    # color = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    labels = ["|GM|=1","|GM|=2","|GM|=3","|GM|=4","|GM|=5","|GM|=6"]
    dim_sum = np.cumsum(dimensions)

    for ii, freq in enumerate(frequencies):
        idx_ir = (ii > dim_sum - 1).sum()
        if (ii in dim_sum-1) and (idx_ir % 2 == 1):
            ax.plot(np.array(distances), freq, label=labels[int(idx_ir//2)], color=color[int(idx_ir//2)])
        else:
            ax.plot(np.array(distances), freq, color=color[int(idx_ir//2)])

    plt.xlabel("qpoints_onedim")
    plt.ylabel("frequencies Thz")
    plt.legend()
    # plt.savefig(path_save_phonon, dpi=600)
    plt.show()



if __name__ == "__main__":
    main()
