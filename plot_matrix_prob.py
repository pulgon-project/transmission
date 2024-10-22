import os.path
import argparse
from ipdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba.cuda.printimpl import print_item
import matplotlib
from pygments import highlight
import matplotlib.patches as patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", help="directory")
    args = parser.parse_args()

    path_directory = args.data_directory
    # path_sym_phonon = os.path.join(path_directory, "sym-adapted-phonon.npz")
    path_irreps_trans = os.path.join(path_directory, "transmission_irreps.npz")
    path_modes_trans = os.path.join(path_directory, "trans_modes.npz")
    path_save = os.path.join(path_directory, "transmission_matrix_irreps.png")

    # phonon_data = np.load(path_sym_phonon)
    irrpes_trans_data = np.load(path_irreps_trans, allow_pickle=True)
    modes_trans_data = np.load(path_modes_trans, allow_pickle=True)

    inc_omega = irrpes_trans_data["inc_omega"]
    matrices_prob = irrpes_trans_data["matrices_prob"]
    Irreps = irrpes_trans_data["Irreps"]



    rows, cols = 4, 5
    multis = len(Irreps) // (rows*cols)
    fsize = 16
    fig, axs = plt.subplots(rows, cols, figsize=(16, 10))
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    for i in range(rows):
        for j in range(cols):
            k = (cols * i + j) * multis
            if matrices_prob[k].size > 0:
                sorted_indices = np.argsort(Irreps[k])
                tmp_max = matrices_prob[k][np.ix_(sorted_indices, sorted_indices)]
                tmp_irreps = np.array(Irreps[k])[sorted_indices]

                un_irreps, counts = np.unique(tmp_irreps, return_counts=True)
                fracs = counts / counts.sum()
                height, width = tmp_max.shape
                im = axs[i, j].matshow(tmp_max, vmin=0.0, vmax=1.0)
                for ii, cot in enumerate(fracs):
                    if ii==0:
                        x0 = -0.5
                        y0 = -0.5
                    else:
                        x0 = x0 + width * fracs[ii-1]
                        y0 = y0 + height * fracs[ii - 1]

                    rect = matplotlib.patches.Rectangle((x0, y0), counts[ii], counts[ii], linewidth=3, edgecolor='red', facecolor='none')
                    axs[i, j].add_patch(rect)

                # axs[i, j].set_xticks(np.arange(len(tmp_irreps)))
                # axs[i, j].set_xticklabels(tmp_irreps)
                arr = np.insert(-0.5 + np.cumsum(counts), 0, -0.5)
                pos_irreps =(arr[:-1] + arr[1:]) / 2
                axs[i, j].set_xticks(pos_irreps)
                axs[i, j].set_xticklabels(un_irreps)

                axs[i, j].set_yticks(pos_irreps)
                axs[i, j].set_yticklabels(un_irreps)
                axs[i, j].tick_params(axis='x', labelsize=14)
                axs[i, j].tick_params(axis='y', labelsize=14)

            else:
                axs[i, j].axis("off")
            axs[i, j].text(
                0.85,
                0.85,
                r"${0:.2f}$".format(inc_omega[k]),
                horizontalalignment="center",
                verticalalignment="center",
                transform=axs[i, j].transAxes,
                fontdict=dict(size=fsize / 1.2, color="red", weight="bold"),
            )
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=1e-3, wspace=1e-3, right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path_save, dpi=500)
    plt.show()


