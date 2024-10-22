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
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", help="directory")
    args = parser.parse_args()
    path_directory = args.data_directory
    path_savefig = os.path.join(path_directory, "phonon-sym_transmission")

    # path_modes1 = os.path.join(path_directory, "trans_modes.npz")
    # trans_mode1 = np.load(path_modes1, allow_pickle=True)
    path_sym_phonon = os.path.join(path_directory, "sym-adapted-phonon.npz")

    path_irreps = os.path.join(path_directory, "transmission_irreps.npz")

    sym_transmission = np.load(path_irreps, allow_pickle=True)

    k_w = sym_transmission["k_w"]
    trans_modes = sym_transmission["trans_modes"]
    inc_omega = sym_transmission["inc_omega"]

    sym_phonon = np.load(path_sym_phonon, allow_pickle=True)

    # trans_values = trans_mode1["trans_values"]

    frequencies = sym_phonon["frequencies"]
    dim_sum = sym_phonon["dim_sum"]
    distances = sym_phonon["distances"]

    # k_w1 = trans_mode1["k_w"]

    # inc_omega = trans_mode1["inc_omega"]
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
        x = k_w[iomega]
        y = omega * np.ones_like(x)
        scatter = plt.scatter(x, y, s=18, c=trans_modes[iomega], cmap="coolwarm", vmin=0, vmax=1, zorder=2)


    plt.colorbar(scatter, label='Values')
    plt.legend(loc="best")
    plt.xlabel("qpoints_onedim")
    plt.ylabel("frequencies * 2 * pi (Thz)")
    plt.savefig(path_savefig, dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
