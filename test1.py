import os.path
import phono3py
from ipdb import set_trace
from ase import Atoms
from pulgon_tools_wip.utils import (
    get_character,
    get_matrices,
    get_matrices_withPhase,
    find_axis_center_of_nanotube,
    brute_force_generate_group_subsquent,
    get_symbols_from_ops
)
from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
from pulgon_tools_wip.detect_generalized_translational_group import CyclicGroupAnalyzer
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol
from utilities import divide_irreps, divide_over_irreps, get_adapted_matrix
import numpy as np
from pulgon_tools_wip.line_group_table import get_family_Num_from_sym_symbol
from pymatgen.core.operations import SymmOp
import phonopy
from ase.io.vasp import read_vasp
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from tqdm import tqdm
from phonopy.units import VaspToTHz
from pulgon_tools_wip.utils import get_character, get_character_num
from itertools import permutations, product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# path_0 = "datas/carbon_nanotube/4x0-u1-3-defect-C-1/phono3py.yaml"
# ph3 = phono3py.load(path_0)
path_0 = "datas/carbon_nanotube/4x0-u1-3-defect-C-1"
# path_0 = "datas/WS2/10x0-u1-3-WMo-C1-1"
# path_0 = "datas/WS2/6-6-u1-3-defect-1"
# path_0 = "datas/WS2/6-6-u1-3-defect-1"
# path_0 = "datas/WS2-MoS2_NNFF-epoch2000-1/6x6-u1-3-WMo-S12-1"
# path_0 = "datas/WS2-MoS2_NNFF-epoch2000-1/10x0-20x0-pristine"
ph_yaml = os.path.join(path_0, "phonopy_pure.yaml")
poscar = os.path.join(path_0, "POSCAR")
img_save = os.path.join(path_0, "img_selection_rules")

if not os.path.exists(img_save):
    os.makedirs(img_save)


ph = phonopy.load(ph_yaml)
# atom = Atoms(cell=ph3.primitive.cell, numbers=ph3.primitive.numbers, positions=ph3.primitive.positions)
atom = read_vasp(poscar)
atom_center = find_axis_center_of_nanotube(atom)

cyclic = CyclicGroupAnalyzer(atom_center, tolerance=1e-2)
obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
nrot = obj.get_rotational_symmetry_number()
aL = atom_center.cell[2, 2]
trans_sym = cyclic.cyclic_group[0]
rota_sym = obj.sch_symbol

family = get_family_Num_from_sym_symbol(trans_sym, rota_sym)
print("family=", family)

set_trace()

trans_op = np.round(cyclic.get_generators(), 5)
rots_op = np.round(obj.get_generators(), 5)
mats = np.vstack(([trans_op], rots_op))
symbols = get_symbols_from_ops(rots_op)

ops, order_ops = brute_force_generate_group_subsquent(mats, symec=1e-2)


ops_car_sym = []
for op in ops:
    tmp_sym = SymmOp.from_rotation_and_translation(
        op[:3, :3], op[:3, 3] * aL
    )
    ops_car_sym.append(tmp_sym)

NQS = 6
# k_start = -np.pi
# k_start = 0
k_start = np.pi
k_end = np.pi

path = [[[0, 0, k_start / 2 / np.pi], [0, 0, k_end / 2 / np.pi]]]
qpoints_ori, connections = get_band_qpoints_and_path_connections(
    path, npoints=NQS
)
qpoints = qpoints_ori[0]
qpoints_1dim = qpoints[:, 2] * 2 * np.pi
qpoints_1dim = qpoints_1dim / aL

ph.run_band_structure(qpoints_ori, path_connections=connections, with_eigenvectors=True)
frequencies, bands, eigvecs_convert = [], [], []
num_atom = len(atom_center.numbers)

for ii, qp in enumerate(tqdm(qpoints_1dim)):  # loop q points
    DictParams = {"qpoints": qp, "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13
    # characters, paras_values, paras_symbols = get_character(DictParams)
    characters, paras_values, paras_symbols = get_character_num(DictParams)
    # modes_IR_idx = [0, 0, 0]
    # res = (characters[modes_IR_idx[0]] * characters[modes_IR_idx[1]] * characters[modes_IR_idx[2]].conj()).sum() / len(ops)

    IR_list = list(range(len(characters)))   # the number of IR
    select_mat = np.zeros((len(characters), len(characters), len(characters)))
    for perm in product(IR_list, repeat=3):
        res = (characters[perm[0]] * characters[perm[1]] * characters[perm[2]].conj()).sum() / len(ops)
        select_mat[perm[0], perm[1], perm[2]] = res
    select_mat1 = (select_mat.real > 1e-1).astype(int)
    x, y, z = np.where(select_mat1 == 1)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(IR_list)):
    #     for j in range(len(IR_list)):
    #         for k in range(len(IR_list)):
    #             if select_mat[i, j, k] == 1:
    #                 ax.bar3d(i, j, k, 1, 1, 1, color='red', alpha=0.3)  # 设置透明度 alpha
    # ax.set_xlabel('IR1')
    # ax.set_ylabel('IR2')
    # ax.set_zlabel('IR3')
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color='red')  # 可以自定义点的大小和颜色
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='IR1',
            yaxis_title='IR2',
            zaxis_title='IR3'
        ),
        title='k=%.2f' % (qp * np.pi)
    )
    fig.show()
    # path_save = os.path.join(img_save, "img_id%d_qp_%.2f.png" % (ii, qp * np.pi))
    # fig.write_image(file=path_save, format='png', scale=3)
    # set_trace()
#     matrices = get_matrices_withPhase(atom_center, ops_car_sym, qp, symprec=1e-2)
#     adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)

#     qz = qpoints[ii]
#     D = ph.get_dynamical_matrix_at_q(qz)
#     D = adapted.conj().T @ D @ adapted
#     start = 0
#     tmp_band, tmp_eigvec = [], []
#     for ir in range(len(dimensions)):
#         end = start + dimensions[ir]
#         block = D[start:end, start:end]
#         eig, eigvecs = np.linalg.eigh(block)
#         e = (
#                 np.sqrt(np.abs(eig))
#                 * np.sign(eig)
#                 * VaspToTHz
#         ).tolist()
#
#         tmp_vec = adapted[:, start:end] @ eigvecs
#         tmp_eigvec.append(tmp_vec)
#         tmp_band.append(e)
#         start = end
#     bands.append(np.concatenate(tmp_band))
#     eigvecs_convert.append(np.concatenate(tmp_eigvec, axis=1))
# bands = np.array(bands)  # .swapaxes(0, 1) # * 2 * np.pi
# eigvecs_convert = np.array(eigvecs_convert)  # .swapaxes(0, 1) # * 2 * np.pi

# ph.band_structure._eigenvalues = bands
# ph.band_structure._eigenvectors = eigvecs_convert
