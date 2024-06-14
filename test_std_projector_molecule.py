import logging
import os.path
import numpy as np
import phonopy
import scipy.linalg as la
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from pymatgen.core.operations import SymmOp
from spglib import get_symmetry_dataset
from pymatgen.util.coord import find_in_coord_list
import scipy
from utilities import commuting

def ir_table():
    GM1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,])

    GM2 = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,])

    GM3 =np.array([np.eye(2),
                  [[0,1],[1,0]],
                  np.eye(2),
                  [[0,1],[1,0]],
                   np.eye(2),
                  [[0, 1], [1, 0]],
                  np.eye(2),
                  [[0, 1], [1, 0]],
                  [[np.exp(1j*2*np.pi/3),0],[0,np.exp(-1j*2*np.pi/3)]],
                  [[0, np.exp(1j * 2 * np.pi / 3)], [np.exp(-1j * 2 * np.pi / 3), 0]],
                  [[np.exp(1j*2*np.pi/3),0],[0,np.exp(-1j*2*np.pi/3)]],
                  [[0, np.exp(1j * 2 * np.pi / 3)], [np.exp(-1j * 2 * np.pi / 3), 0]],
                  [[np.exp(1j * 2 * np.pi / 3), 0], [0, np.exp(-1j * 2 * np.pi / 3)]],
                  [[0, np.exp(1j * 2 * np.pi / 3)], [np.exp(-1j * 2 * np.pi / 3), 0]],
                  [[np.exp(1j * 2 * np.pi / 3), 0], [0, np.exp(-1j * 2 * np.pi / 3)]],
                  [[0, np.exp(1j * 2 * np.pi / 3)], [np.exp(-1j * 2 * np.pi / 3), 0]],
                  [[np.exp(-1j * 2 * np.pi / 3), 0], [0, np.exp(1j * 2 * np.pi / 3)]],
                  [[0, np.exp(-1j * 2 * np.pi / 3)], [np.exp(1j * 2 * np.pi / 3), 0]],
                  [[np.exp(-1j * 2 * np.pi / 3), 0], [0, np.exp(1j * 2 * np.pi / 3)]],
                  [[0, np.exp(-1j * 2 * np.pi / 3)], [np.exp(1j * 2 * np.pi / 3), 0]],
                  [[np.exp(-1j * 2 * np.pi / 3), 0], [0, np.exp(1j * 2 * np.pi / 3)]],
                  [[0, np.exp(-1j * 2 * np.pi / 3)], [np.exp(1j * 2 * np.pi / 3), 0]],
                  [[np.exp(-1j * 2 * np.pi / 3), 0], [0, np.exp(1j * 2 * np.pi / 3)]],
                  [[0, np.exp(-1j * 2 * np.pi / 3)], [np.exp(1j * 2 * np.pi / 3), 0]],]
                  )

    GM4 =np.array([np.eye(3),
                  [[-1,0,0],[0,0,1],[0,-1,0]],
                    [[1,0,0],[0,-1,0],[0,0,-1]],
                  [[-1,0,0],[0,0,-1],[0,1,0]],
                    [[-1,0,0],[0,1,0],[0,0,-1]],
                  [[1,0,0],[0,0,1],[0,1,0]],
                  [[-1,0,0],[0,-1,0],[0,0,1]],
                  [[1,0,0],[0,0,-1],[0,-1,0]],
                  [[0,0,1],[1,0,0],[0,1,0]],
                  [[0,-1,0],[-1,0,0],[0,0,1]],
                  [[0,0,-1],[1,0,0],[0,-1,0]],
                  [[0,1,0],[-1,0,0],[0,0,-1]],
                  [[0,0,-1],[-1,0,0],[0,1,0]],
                  [[0,1,0],[1,0,0],[0,0,1]],
                  [[0,0,1],[-1,0,0],[0,-1,0]],
                  [[0,-1,0],[1,0,0],[0,0,-1]],
                  [[0,1,0],[0,0,1],[1,0,0]],
                  [[0,0,1],[0,-1,0],[-1,0,0]],
                  [[0,-1,0],[0,0,-1],[1,0,0]],
                  [[0,0,-1],[0,1,0],[-1,0,0]],
                  [[0,1,0],[0,0,-1],[-1,0,0]],
                  [[0,0,1],[0,1,0],[1,0,0]],
                  [[0,-1,0],[0,0,1],[-1,0,0]],
                  [[0,0,-1],[0,-1,0],[1,0,0]]]
                  )

    GM5 =np.array([np.eye(3),
                  [[1,0,0],[0,0,-1],[0,1,0]],
                  [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                  [[1,0,0],[0,0,1],[0,-1,0]],
                    [[-1,0,0],[0,1,0],[0,0,-1]],
                  [[-1,0,0],[0,0,-1],[0,-1,0]],
                  [[-1,0,0],[0,-1,0],[0,0,1]],
                  [[-1,0,0],[0,0,1],[0,1,0]],
                  [[0,0,1],[1,0,0],[0,1,0]],
                  [[0,1,0],[1,0,0],[0,0,-1]],
                  [[0,0,-1],[1,0,0],[0,-1,0]],
                  [[0,-1,0],[1,0,0],[0,0,1]],
                  [[0,0,-1],[-1,0,0],[0,1,0]],
                  [[0,-1,0],[-1,0,0],[0,0,-1]],
                  [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                  [[0,1,0],[-1,0,0],[0,0,1]],
                  [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                  [[0,0,-1],[0,1,0],[1,0,0]],
                  [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                  [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
                  [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                  [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
                  [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                  [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],]
                  )
    res = [GM1, GM2, GM3, GM4, GM5]
    return res


def get_perms_from_ops_mol(atoms, ops_sym, symprec=1e-2, round=4):
    """get the permutation table from symmetry operations

    Args:
        atoms:
        symprec:

    Returns: permutation table
    """
    natoms = len(atoms.numbers)
    coords_scaled = atoms.get_scaled_positions()
    coords_scaled_center = np.remainder(
        np.round(coords_scaled, round) - [0.5, 0.5, 0.5], [1, 1, 1]
    )

    perms = []
    for ii, op in enumerate(ops_sym):
        tmp_perm = np.zeros((1, len(atoms.numbers)))[0]
        for jj, site in enumerate(atoms):
            pos = (site.scaled_position - [0.5, 0.5, 0.5]) @ atoms.cell

            tmp = op.operate(pos)
            tmp1 = np.remainder(
                np.round(tmp @ np.linalg.inv(atoms.cell), round), [1, 1, 1]
            )
            idx2 = find_in_coord_list(coords_scaled_center, tmp1, symprec)

            if idx2.size == 0:
                set_trace()
                logging.ERROR("tolerance exceed while calculate perms")
            tmp_perm[jj] = idx2

        idx = len(np.unique(tmp_perm))
        if idx != natoms:
            logging.ERROR("perms numebr != natoms")
        perms.append(tmp_perm)
    perms_table = np.array(perms).astype(np.int32)
    return perms_table


def get_matrices(atoms, ops_sym):
    perms_table = get_perms_from_ops_mol(atoms, ops_sym)
    natoms = len(atoms.numbers)
    matrices = []
    for ii, perm in enumerate(perms_table):
        matrix = np.zeros((3 * natoms, 3 * natoms))
        for jj in range(natoms):
            idx = perm[jj]
            matrix[3 * idx : 3 * (idx + 1), 3 * jj : 3 * (jj + 1)] = ops_sym[
                ii
            ].rotation_matrix.copy()
        matrices.append(matrix)
    return matrices


def get_modified_projector_of_molecular(g_rot, atom):
    matrices_apg = get_matrices(atom, g_rot)
    GM = ir_table()

    basis, dimensions = [], []
    for i_Dmu, Dmu_rot in enumerate(GM):

        # the degeneracy of IR
        if Dmu_rot[0].ndim==0:
            d_mu = 1
        else:
            d_mu = len(Dmu_rot[0])

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
            set_trace()
        # print("m=%s" % tmp_m1, "error=%s" % error)
        basis.append(u[:,:num_modes])
        dimensions.append(num_modes)

    adapted = np.concatenate(basis, axis=1)
    if adapted.shape[0] != adapted.shape[1]:
        print("the number of eigenvector is %d" % adapted.shape[1], "%d" % adapted.shape[0] + "is required")
    return adapted, dimensions


def main():
    path_0 = "datas/molecular/CH4"
    path_yaml = os.path.join(path_0, "phonopy_disp.yaml")
    path_fc_set = os.path.join(path_0, "FORCE_SETS")
    path_save_fc_sym = os.path.join(path_0, "fc_sym_std")
    path_save_fc_adapted = os.path.join(path_0, "adapted_std")

    atom = read_vasp(os.path.join(path_0, "H4C"))
    datasets = get_symmetry_dataset(atom)
    rots = datasets["rotations"]
    trans = datasets["translations"]

    sym = []
    for ii, rot in enumerate(rots):
        sym.append(SymmOp.from_rotation_and_translation(rotation_matrix=rot, translation_vec=trans[ii]))
    adapted, dimensions = get_modified_projector_of_molecular(sym, atom)

    ##### force constant #########
    phonon = phonopy.load(phonopy_yaml=path_yaml, force_sets_filename=path_fc_set)
    fc = phonon.force_constants
    fc_reshape = fc.transpose(0,2,1,3).reshape(15,15)
    fc_adapted = np.abs(adapted.conj().T @ fc_reshape @ adapted)
    ##############################

    np.savetxt(path_save_fc_sym, fc_adapted, fmt="%10.3f")
    np.savetxt(path_save_fc_adapted, adapted)

    print("dimension:")
    print(dimensions)


if __name__ == '__main__':
    main()

