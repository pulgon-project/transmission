import numpy as np
import os
from ipdb import set_trace
from ase.units import _hplanck, Bohr, kB, eV, J
from ase.io.vasp import read_vasp
import matplotlib.pyplot as plt


def load_data():
    path_0 = "datas/WS2-MoS2_NNFF-epoch2000-1"

    sym1 = "C10v"
    sym2 = "C1"
    ind1 = 5
    ind2 = 5
    chariality = "10x0-20x0"
    num_cell = 3
    element = "MoW"


    path_save = os.path.join(path_0, "thermal_conductivity-%s-%s-%d-C1-%d" % (element, sym1, ind1, ind2))
    path_st1 = os.path.join(path_0,'%s-u1-%d-defect-%s-%s-%d' % (chariality, num_cell, element, sym1, ind1))
    path_st2 = os.path.join(path_0,'%s-u1-%d-defect-%s-%s-%d' % (chariality, num_cell, element, sym2, ind2))

    # path_0 = "datas/carbon_nanotube"
    # ind1 = 2
    # ind2 = 2
    # path_save = os.path.join(path_0, "thermal_conductivity-C5v-%d-C1-%d" % (ind1, ind2))
    # path_st1 = os.path.join(path_0,'5x0-10x0-u1-3-defect-C5v-%d' % ind1)
    # path_st2 = os.path.join(path_0,'5x0-10x0-u1-3-defect-C1-%d' % ind2)

    path_trans1 = os.path.join(path_st1 , "transmission_irreps.npz")
    path_trans2 = os.path.join(path_st2,  "transmission_irreps.npz")

    path_poscar1 = os.path.join(path_st1, "POSCAR")
    path_poscar2 = os.path.join(path_st2, "POSCAR")

    data1 = np.load(path_trans1)
    data2 = np.load(path_trans2)

    inc_omega = data1["inc_omega"]
    trans1 = data1["trans"]
    trans2 = data2["trans"]
    # NLp_irreps1 = data1["NLp_irreps"]
    # NLp_irreps2 = data2["NLp_irreps"]

    atoms1 = read_vasp(path_poscar1)
    atoms2 = read_vasp(path_poscar2)
    L1 = atoms1.cell[2,2] * 1e-10
    L2 = atoms2.cell[2,2] * 1e-10

    max_rad1 = max( np.sqrt(np.sum(((atoms1.get_scaled_positions() - [0.5,0.5,0]) @ atoms1.cell)[:,:2] ** 2, axis=1)) ) * 1e-10
    max_rad2 = max( np.sqrt(np.sum(((atoms2.get_scaled_positions() - [0.5,0.5,0]) @ atoms2.cell)[:,:2] ** 2, axis=1)) ) * 1e-10
    area1 = np.pi * (max_rad1 ** 2)
    area2 = np.pi * (max_rad2 ** 2)
    return inc_omega, trans1, trans2, area1, area2, L1, L2, path_save, sym1, sym2


def main():
    inc_omega, trans1, trans2, area1, area2, L1, L2, path_save, sym1, sym2 = load_data()
    hbar = _hplanck / np.pi / 2
    delta_freq = (inc_omega[-1] - inc_omega[-2]) * 1e12

    delta_energy = hbar * delta_freq
    kB_j = kB * eV / J

    k1, k2 = [], []
    g1, g2 = [], []
    T = np.linspace(1,500, 100)
    # T = [1000]
    for t in T:
        tmp_k1, tmp_k2 = 0, 0
        tmp_g1, tmp_g2 = 0, 0

        contribute_g1 = []
        contribute_g2 = []
        E_n0_T = []

        for ii, freq in enumerate(inc_omega):
            freq = freq / 2 / np.pi * 1e12
            # freq = freq * 10 ** 12

            E_n0_T.append((hbar * freq * freq * hbar * np.exp(hbar*freq/kB_j/t) / (kB_j*(t**2)*((np.exp(hbar*freq/kB_j/t)-1)**2))))
            tmp_g1 += (hbar * freq * trans1[ii]  * freq * hbar * np.exp(hbar*freq/kB_j/t) / (kB_j*(t**2)*((np.exp(hbar*freq/kB_j/t)-1)**2))) * delta_energy
            tmp_g2 += (hbar * freq * trans2[ii]  * freq * hbar * np.exp(hbar*freq/kB_j/t) / (kB_j*(t**2)*((np.exp(hbar*freq/kB_j/t)-1)**2))) * delta_energy
            contribute_g1.append((hbar * freq * trans1[ii]  * freq * hbar * np.exp(hbar*freq/kB_j/t) / (kB_j*(t**2)*((np.exp(hbar*freq/kB_j/t)-1)**2))) * delta_energy)
            contribute_g2.append((hbar * freq * trans2[ii]  * freq * hbar * np.exp(hbar*freq/kB_j/t) / (kB_j*(t**2)*((np.exp(hbar*freq/kB_j/t)-1)**2))) * delta_energy)

            # tmp_k1 += (trans1[ii] * (freq ** 2) * hbar * np.exp(hbar*freq/kB_j/t) / (kB_j*(t**2)*((np.exp(hbar*freq/kB_j/t)-1)**2))) * delta_freq
            # tmp_k2 += (trans2[ii] * (freq ** 2) * hbar * np.exp(hbar*freq/kB_j/t) / (kB_j*(t**2)*((np.exp(hbar*freq/kB_j/t)-1)**2))) * delta_freq
        # tmp_g1 = tmp_g1 / 2 / np.pi / hbar / area1
        # tmp_g2 = tmp_g2 / 2 / np.pi / hbar / area2

        contribute_g1 = np.array(contribute_g1) / 2 / np.pi / hbar
        contribute_g2 = np.array(contribute_g2) / 2 / np.pi / hbar

        tmp_g1 = tmp_g1 / 2 / np.pi / hbar
        tmp_g2 = tmp_g2 / 2 / np.pi / hbar
        # tmp_k1 = tmp_k1 * hbar * L1 / area1 / 2 / np.pi
        # tmp_k2 = tmp_k2 * hbar * L2 / area2 / 2 / np.pi

        tmp_k1 = tmp_k1 * hbar * L1 / 2 / np.pi
        tmp_k2 = tmp_k2 * hbar * L2 / 2 / np.pi
        g1.append(tmp_g1)
        g2.append(tmp_g2)
        # tmp_k1 = L1/ tmp_g1
        # tmp_k2 = L2/ tmp_g2
        k1.append(tmp_k1)
        k2.append(tmp_k2)
    # fig1, axes1 = plt.subplots()
    # plt.plot(T, k1, label="C10v")
    # plt.plot(T, k2, label="C1")
    # plt.xlim(left=0.0)
    # plt.ylim(bottom=0.0)
    # plt.legend(loc="best")
    # # plt.tight_layout()
    # plt.xlabel("T (K)")
    # plt.ylabel("Thermal conductances " + r"$(W \cdot m ^{-3} \cdot K^{-1})$")
    fig2, axes2 = plt.subplots()
    plt.plot(T, g1, label="%s" % sym1)
    plt.plot(T, g2, label="%s" % sym2)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.legend(loc="best")
    # plt.tight_layout()
    plt.xlabel("T (K)")
    plt.ylabel("Thermal conductances " + r"$(K\cdot W^{-1})$")
    plt.tick_params(labelsize=14)

    # fig3, axes3 = plt.subplots()
    # plt.plot(inc_omega, contribute_g1, label="C11v")
    # plt.plot(inc_omega, contribute_g2, label="C1")
    # plt.xlim(left=0.0)
    # plt.ylim(bottom=0.0)
    # plt.legend(loc="best")
    # # plt.tight_layout()
    # plt.xlabel("frequencies ")
    # plt.ylabel("Thermal conductances " + r"$(K\cdot W^{-1})$")
    # plt.tick_params(labelsize=14)
    # fig4, axes4 = plt.subplots()
    # plt.plot(inc_omega, E_n0_T)
    # plt.xlabel("frequencies ")
    # plt.ylabel(r"$E \cdot \partial n_{0}(\omega, T) / \partial T$")

    plt.savefig(path_save, dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
