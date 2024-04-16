import logging
import os.path
import matplotlib.pyplot as plt
import numpy as np
import phonopy
import pretty_errors
from ipdb import set_trace
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from tqdm import tqdm


def main():
    path_0 = "datas/WS2/6-6-u1-3-defect-1"
    path_yaml = os.path.join(path_0, "phonopy_pure.yaml")
    path_fc_continum = os.path.join(path_0, "FORCE_CONSTANTS_pure.continuum")
    path_save_phonon = os.path.join(path_0, "phonon_defect_raw")
    path_save_phonon_transmission = os.path.join(path_0, "transmission_defect_from_phonon")

    phonon = phonopy.load(phonopy_yaml=path_yaml, force_constants_filename=path_fc_continum, is_compact_fc=True)

    NQS = 51
    k_start = -np.pi
    k_end = np.pi

    path = [[[0, 0, k_start/2/np.pi], [0, 0, k_end/2/np.pi]]]
    qpoints, connections = get_band_qpoints_and_path_connections(
        path, npoints=NQS
    )
    qpoints = qpoints[0]

    # %%
    frequencies_raw, distances = [], []
    for ii, qz in enumerate(tqdm(qpoints)):
        D = phonon.get_dynamical_matrix_at_q(qz)
        eigvals, eigvecs = np.linalg.eigh(D)
        eigvals = eigvals.real
        frequencies_raw.append(np.sqrt(abs(eigvals)) * np.sign(eigvals) * VaspToTHz)
        set_trace()


        if ii == 0:
            tmp_dis = np.dot(qz, phonon.supercell.get_cell())[2]
            distances.append(tmp_dis)
            q_last = qz.copy()
        else:
            tmp_dis += np.linalg.norm(np.dot(qz - q_last, phonon.supercell.get_cell()))
            distances.append(tmp_dis)
            q_last = qz.copy()
    frequencies_raw = np.array(frequencies_raw).T

    #### raw phonon
    fig, ax = plt.subplots()
    for i, f_raw in enumerate(frequencies_raw):
        if i == 0:
            ax.plot(distances, f_raw, 'o-', markersize=3, color="grey", label="1S defect")
        else:
            ax.plot(distances, f_raw, 'o-',markersize=3, color="grey")

    plt.xlabel("distances")
    plt.ylabel("frequencies Thz")
    plt.legend()
    # plt.savefig(path_save_phonon, dpi=600)

    plt.show()

if __name__ == "__main__":
    main()
