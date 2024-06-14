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
    brute_force_generate_group_subsquent,
)
from pymatgen.core.operations import SymmOp
from tqdm import tqdm
from utilities import counting_y_from_xy, get_adapted_matrix, commuting
import decimation
from spglib import get_symmetry_dataset

def main():
    path_0 = "datas/WS2/6-6-u1-3-defect-1"
    # path_0 = "datas/WS2/6-6-u2-5-defect-1"
    # path_0 = "datas/carbon_nanotube/4x1-u1-3-defect-C-1"
    # path_0 = "datas/carbon_nanotube/4x0-u1-3-defect-C-1"
    path_yaml = os.path.join(path_0, "phonopy_pure.yaml")
    path_fc_continum = os.path.join(path_0, "FORCE_CONSTANTS_pure.continuum")
    path_save_phonon = os.path.join(path_0, "phonon_defect_sym_adapted")

    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)
    poscar_phonopy = phonon.primitive
    poscar_ase = Atoms(cell=poscar_phonopy.cell, positions=poscar_phonopy.positions, numbers=poscar_phonopy.numbers)

    cyclic = CyclicGroupAnalyzer(poscar_ase, tolerance=1e-2)
    aL = poscar_ase.cell[2,2]
    atom = cyclic._atom
    atom_center = find_axis_center_of_nanotube(atom)
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)

    NQS = 21
    k_start = -np.pi + 0.1
    k_end = np.pi - 0.1

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    qpoints = qpoints[0]
    qpoints_1dim = qpoints[:,2] * 2 * np.pi
    qpoints_1dim = qpoints_1dim / aL

    # path_LR_blocks = os.path.join(path_0, "pure_fc.npz")
    # LR_blocks = np.load(path_LR_blocks)
    # HL = LR_blocks["H00"]
    # TL = LR_blocks["H01"]
    # mass_L = np.diag(np.power(ase.data.atomic_masses[poscar_ase.numbers], -1/2))
    # unit = 1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J
    # # unit = 1
    # HL1 = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, HL), mass_L)
    # HL = np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, HL), mass_L)
    # TL1 = unit * np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, TL), mass_L)
    # TL = np.einsum('ijlm,jk->iklm',np.einsum('ij,jkln->ikln',mass_L, TL), mass_L)
    # HL1 = HL1.transpose((0, 2, 1, 3)).reshape((HL1.shape[0] * 3, -1))
    # HL = HL.transpose((0, 2, 1, 3)).reshape((HL.shape[0] * 3, -1))
    # TL1 = TL1.transpose((0, 2, 1, 3)).reshape((TL1.shape[0] * 3, -1))
    # TL = TL.transpose((0, 2, 1, 3)).reshape((TL.shape[0] * 3, -1))
    # qvec = np.linspace(k_start, k_end, num=NQS)
    # omegaL, vgL = decimation.q2omega(HL1, TL1, qvec)

    ################ family 4 ##################
    family = 4
    obj = LineGroupAnalyzer(atom_center, tolerance=1e-2)
    nrot = obj.get_rotational_symmetry_number()
    sym  = []
    tran = SymmOp.from_rotation_and_translation(Cn(2*nrot), [0, 0, 1/2])
    rots = SymmOp.from_rotation_and_translation(Cn(nrot), [0, 0, 0])
    mirror = SymmOp.reflection([0,0,1], [0,0,0.25])
    # sym.append(tran.affine_matrix)
    sym.append(rots.affine_matrix)
    sym.append(mirror.affine_matrix)
    ops, order_ops = brute_force_generate_group_subsquent(sym, symec=1e-5)

    res2 = []
    for tmp1 in order_ops:
        res1 = []
        for tmp2 in tmp1:
            if tmp2!=0:
                res1.append(tmp2+1)
            else:
                res1.append(tmp2)
        res2.append(res1)
    order_ops = res2
    # set_trace()
    ###############################################
    ops_car_sym = []
    for op in ops:
        tmp_sym1 = SymmOp.from_rotation_and_translation(
            op[:3, :3], op[:3, 3] * aL
        )
        ops_car_sym.append(tmp_sym1)
    matrices = get_matrices(atom_center, ops_car_sym)

    frequencies, distances, bands = [], [], []
    num_atom = len(poscar_ase.numbers)
    for ii, qp in enumerate(tqdm(qpoints_1dim)):   # loop q points
        DictParams = {"qpoints":qp,  "nrot": nrot, "order": order_ops, "family": family, "a": aL}  # F:2,4, 13
        adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)
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
        D = phonon.get_dynamical_matrix_at_q(q)
        # qp = qpoints_1dim[ii]
        # D = TL.conj().transpose() * np.exp(-1j*qp*aL) + HL + TL * np.exp(1j*qp*aL)

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


    #### symmetry adapted
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'yellow', 'pink', 'olive', 'sage', 'slategray', 'darkkhaki', 'yellowgreen']
    # color = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    labels = ["|m|=0","|m|=1","|m|=2","|m|=3","|m|=4","|m|=5","|m|=6","|m|=7","|m|=8","|m|=9", "|m|=10", "|m|=11","|m|=12", "|m|=13", "|m|=14"]
    dim_sum = np.cumsum(dimensions)

    if family==4:
        for ii, freq in enumerate(frequencies):
            idx_ir = (ii > dim_sum - 1).sum()
            if ii in dim_sum-1 and idx_ir>=nrot - 1:
                ax.plot(np.array(distances), freq, label=labels[int(abs(idx_ir-nrot+1))], color=color[int(abs(idx_ir-nrot+1))])
            else:
                ax.plot(np.array(distances), freq, color=color[int(abs(idx_ir-nrot+1))])

    plt.xlabel("qpoints_onedim")
    plt.ylabel("frequencies Thz")
    plt.legend()
    plt.savefig(path_save_phonon, dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
