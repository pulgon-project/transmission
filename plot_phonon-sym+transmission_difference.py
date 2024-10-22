import os.path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from ipdb import set_trace
import os
from ase.io.vasp import read_vasp
from ipdb import set_trace


def main():
    # path_0 = "datas/carbon_nanotube"
    # ind1 = 2
    # ind2 = 2
    # sym1 = "C5v-" + str(ind1)
    # sym2 = "C1-" + str(ind2)
    # path_1 = os.path.join(path_0, '5x0-10x0-u1-3-defect-%s' % sym1)
    # path_2 = os.path.join(path_0, '5x0-10x0-u1-3-defect-%s' % sym2)

    path_0 = "datas/WS2-MoS2"
    ind1 = 2
    ind2 = 2
    sym1 = "C11v-" + str(ind1)
    sym2 = "C1-" + str(ind2)
    charality = "11x0-22x0"
    num_cell = 3

    path_1 = os.path.join(path_0, '%s-u1-%d-defect-S-%s' % (charality, num_cell, sym1))
    path_2 = os.path.join(path_0, '%s-u1-%d-defect-S-%s' % (charality, num_cell, sym2))

    path_savefig = os.path.join(path_0, "phonon_trans_difference-%s-%s" % (sym1, sym2))


    path_modes1 = os.path.join(path_1, "transmission_irreps.npz")
    path_modes2 = os.path.join(path_2, "transmission_irreps.npz")
    path_sym_phonon = os.path.join(path_1, "sym-adapted-phonon.npz")


    trans_mode1 = np.load(path_modes1, allow_pickle=True)
    trans_mode2 = np.load(path_modes2, allow_pickle=True)
    sym_phonon = np.load(path_sym_phonon, allow_pickle=True)

    trans_values1 = trans_mode1["trans_modes"]
    trans_values2 = trans_mode2["trans_modes"]
    trans_values = trans_values2 - trans_values1

    frequencies = sym_phonon["frequencies"]
    dim_sum = sym_phonon["dim_sum"]
    distances = sym_phonon["distances"]

    k_w1 = trans_mode1["k_w"]
    k_w2 = trans_mode2["k_w"]

    inc_omega = trans_mode1["inc_omega"]
    # qpoints_onedim = trans_mode1["qpoints_onedim"]
    # omegaL = trans_mode1["omegaL"]

    fig, ax = plt.subplots(figsize=(16, 10))
    family = 6
    color = ['k', 'green', 'Olive','slategray', 'purple', 'magenta',  'cyan','blue', 'red', 'yellow', 'pink',  'darkkhaki', 'yellowgreen']
    labels = ["|m|=0","|m|=1","|m|=2","|m|=3","|m|=4","|m|=5","|m|=6","|m|=7","|m|=8","|m|=9", "|m|=10", "|m|=11","|m|=12", "|m|=13", "|m|=14"]
    if family==6:
        for ii, freq in enumerate(frequencies):
            idx_ir = (ii > dim_sum - 1).sum()
            if ii in dim_sum-1:
                ax.plot(distances, freq, label=labels[int(abs(idx_ir))], color=color[int(abs(idx_ir))], zorder=1)
            else:
                ax.plot(distances, freq, color=color[int(abs(idx_ir))], zorder=1)
    for iomega, omega in enumerate(inc_omega):
        x = k_w1[iomega]
        y = omega * np.ones_like(x)
        scatter = plt.scatter(x, y, s=20, c=trans_values[iomega], cmap="coolwarm", vmin=-1, vmax=1, zorder=2)

    plt.colorbar(scatter, label='Values')
    plt.legend(loc="best")
    plt.xlabel("qpoints_onedim")
    plt.ylabel("frequencies * 2 * pi (Thz)")
    # plt.title(r"$T_{C1}-T_{C5v}$")
    plt.title(r"$T_{C1}-T_{C10v}$")

    plt.savefig(path_savefig, dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
