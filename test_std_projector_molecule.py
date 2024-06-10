import copy
import logging
import os.path
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import pretty_errors
import scipy.linalg as la
from ase.io.vasp import read_vasp, write_vasp
from ipdb import set_trace
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.utils import (
    fast_orth,
    get_matrices,
)
from pymatgen.core.operations import SymmOp
from tqdm import tqdm
from utilities import counting_y_from_xy, get_adapted_matrix
import decimation
import ase
from spglib import get_symmetry_dataset
from pymatgen.util.coord import find_in_coord_list
from sympy.physics.quantum import TensorProduct
import scipy


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
        tmp1, tmp2 = [], []
        for ii in range(len(Dmu_rot)):
            if d_mu == 1:
                num_modes += Dmu_rot[ii] * matrices_apg[ii].trace()
                if ii == 0:
                    projector = Dmu_rot[ii].conj() * matrices_apg[ii]
                else:
                    projector += Dmu_rot[ii].conj() * matrices_apg[ii]
            else:
                num_modes += Dmu_rot[ii].trace() * matrices_apg[ii].trace()
                if ii == 0:
                    projector = Dmu_rot[ii].conj().trace() * matrices_apg[ii]
                else:
                    projector += Dmu_rot[ii].conj().trace() * matrices_apg[ii]

        num_modes = int(num_modes.real * d_mu / len(Dmu_rot))
        projector = d_mu * projector / (len(Dmu_rot))
        # set_trace()

        if num_modes ==0:
            dimensions.append(num_modes)
            continue

        u, s, vh = scipy.linalg.svd(projector)

        # set_trace()
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
    atom = read_vasp(os.path.join(path_0, "H4C.vasp"))
    datasets = get_symmetry_dataset(atom)
    rots = datasets["rotations"]
    trans = datasets["translations"]

    sym = []
    for ii, rot in enumerate(rots):
        sym.append(SymmOp.from_rotation_and_translation(rotation_matrix=rot, translation_vec=trans[ii]))
    adapted, dimensions = get_modified_projector_of_molecular(sym, atom)
    set_trace()


if __name__ == '__main__':
    main()

