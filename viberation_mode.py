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
    get_matrices_withPhase,
    find_axis_center_of_nanotube,
    Cn,
    S2n,
    brute_force_generate_group_subsquent,
    get_symbols_from_ops
)
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol

from pymatgen.core.operations import SymmOp
from tqdm import tqdm

from utilities import  get_adapted_matrix
import decimation
from spglib import get_symmetry_dataset
import ase
import argparse
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation



def plot_vibration_3d(atomic_positions, frequencies, eigenvectors, mode_index=0, amplitude=0.1, frames=100):
    eigenvector = eigenvectors[mode_index].real  # 提取实部
    frequency = frequencies[mode_index]
    omega = 2 * np.pi * frequency  # 振动角频率

    num_atoms = len(atomic_positions)

    # 初始化 3D 图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    ax.set_title(f'Γ-point Mode {mode_index+1}, Frequency: {frequency:.2f} THz')

    # 绘制原子
    atoms = ax.scatter([], [], [], c='b', s=50)


    # 绘制振动方向箭头
    arrows = []
    for pos in atomic_positions:
        arrow = ax.quiver(pos[0], pos[1], pos[2], 0, 0, 0, color='r', length=0.2, normalize=True)
        arrows.append(arrow)

    # 初始化函数
    def init():
        atoms._offsets3d = ([], [], [])  # 将初始数据设为空
        return atoms, *arrows

    # 更新函数
    def update(frame):
        t = frame / frames
        displacements = amplitude * np.sin(omega * t) * eigenvector.reshape(-1, 3)
        new_positions = atomic_positions + displacements

        print(sum(abs(displacements)))

        atoms._offsets3d = (new_positions[:, 0], new_positions[:, 1], new_positions[:, 2])


        # 更新箭头
        for arrow, pos, disp in zip(arrows, atomic_positions, displacements):
            # 每次更新箭头的方向，而不是删除
            arrow.remove()  # 删除旧箭头
            new_arrow = ax.quiver(pos[0], pos[1], pos[2], disp[0], disp[1], disp[2], color='r', length=0.3, normalize=True)
            arrows[arrows.index(arrow)] = new_arrow  # 更新箭头列表

        return atoms, *arrows

    # 动画
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=50)
    plt.show()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "data_directory", help="directory")
    parser.add_argument(
        "-k",
        "--kpoints",
        type=int,
        default=11,
        help="plot the raw phonon or not",
    )

    args = parser.parse_args()
    path_0 = args.data_directory
    num_k = args.kpoints

    path_yaml = os.path.join(path_0, "phonopy_pure.yaml")
    path_fc_continum = os.path.join(path_0, "FORCE_CONSTANTS_pure.continuum")
    path_save_phonon = os.path.join(path_0, "phonon_defect_sym_adapted")
    path_savedata = os.path.join(path_0, "sym-adapted-phonon")

    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)
    # phonon = phonopy.load(phonopy_yaml=path_yaml, is_compact_fc=True)
    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)

    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    aL = poscar_ase.cell[2,2]
    atom = cyclic._atom
    atom_center = find_axis_center_of_nanotube(atom)

    NQS = num_k
    k_start = -np.pi
    # k_start = 0
    k_end = np.pi

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    qpoints = qpoints[0]
    qpoints_1dim = qpoints[:,2] * 2 * np.pi
    qpoints_1dim = qpoints_1dim / aL


    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    aL = atom_center.cell[2, 2]
    trans_sym = cyclic.cyclic_group[0]
    rota_sym = obj.sch_symbol

    family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)


    trans_op = cyclic.get_generators()
    rots_op = obj.get_generators()
    mats = [trans_op] + rots_op
    symbols = get_symbols_from_ops(rots_op)

    # ################ family 8 ###################
    family = 8
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    sym  = []
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    mirror = SymmOp.reflection([1,0,0], [0,0,0])
    sym.append(tran.affine_matrix)
    sym.append(rots.affine_matrix)
    sym.append(mirror.affine_matrix)
    # ############### family 6 ######################
    # family = 6
    # obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    # nrot = obj.rot_sym[0][1]
    # sym  = []
    # rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    # # mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    # # sym.append(tran.affine_matrix)
    # sym.append(rots.affine_matrix)
    # sym.append(obj.get_generators()[1])
    # ################################################
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-4)
    ops_car_sym = []
    for op in ops:
        tmp_sym = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym)

    num_atom = len(poscar_ase.numbers)

    ind_qp = 0
    qp = qpoints_1dim[ind_qp]
    qz = qpoints[ind_qp]

    # DictParams = {"qpoints":qp,  "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13
    # matrices = get_matrices_withPhase(atom_center, ops_car_sym, qp)
    # adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)

    D = phonon.get_dynamical_matrix_at_q(qz)
    # D = adapted.conj().T @ D @ adapted
    # start = 0
    # tmp_band = []
    # for ir in range(len(dimensions)):
    #     end = start + dimensions[ir]
    #     block = D[start:end, start:end]
    #     eig, eigvecs = np.linalg.eigh(block)
    #     e = (
    #             np.sqrt(np.abs(eig))
    #             * np.sign(eig)
    #             * VaspToTHz
    #     ).tolist()
    #     set_trace()
    #     tmp_band.append(e)
    #     start = end
    # bands = np.concatenate(tmp_band)

    eigvals, eigvecs = np.linalg.eigh(D)
    eigvals = eigvals.real
    frequencies = np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz
    atomic_positions = poscar_ase.positions
    plot_vibration_3d(atomic_positions, frequencies, eigvecs, mode_index=0, amplitude=0.1, frames=100)


if __name__ == "__main__":
    main()
