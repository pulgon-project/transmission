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


path_0 = "datas/WS2-MoS2_NNFF-epoch2000-1"
sym1 = "C10v"
sym2 = "C1"
ind1 = 5
ind2 = 5
label1 = r"$C_{10v}$"
label2 = r"$C_{1}$"
chariality = "10x0-20x0"
element = "MoW"

num_cell = 3

path_1 = os.path.join(path_0, '%s-u1-%d-defect-%s-%s-%d' % (chariality,num_cell, element, sym1,ind1))
path_2 = os.path.join(path_0, '%s-u1-%d-defect-%s-%s-%d' % (chariality,num_cell, element, sym2,ind2))

# path_0 = "datas/carbon_nanotube"
# ind1 = 6
# ind2 = 4
# sym1 = "C5v-" + str(ind1)
# sym2 = "C1-" + str(ind2)
# path_1 = os.path.join(path_0, '5x0-10x0-u1-3-defect-%s' % sym1)
# path_2 = os.path.join(path_0, '5x0-10x0-u1-3-defect-%s' % sym2)
# savename1 = r"$C_{5v}$"
# savename2 = r"$C_{1}$"


path_savefig = os.path.join(path_0, "Irreps-%s-%s-%d-%s-%d" % (element,sym1, ind1, sym2, ind2))

path_trans1 = os.path.join(path_1, "transmission_irreps.npz")
path_trans2 = os.path.join(path_2, "transmission_irreps.npz")


data1 = np.load(path_trans1)
data2 = np.load(path_trans2)


inc_omega = data1["inc_omega"]
trans1 = data1["trans"]
trans2 = data2["trans"]
NLp_irreps1 = data1["NLp_irreps"]
NLp_irreps2 = data2["NLp_irreps"]


colors = [value for key, value in mcolors.XKCD_COLORS.items()]
# color_map = cm.get_cmap('viridis_r')
# color_map = plt.get_cmap('tab10')
color_map = plt.get_cmap('Set1')


norm = plt.Normalize(0, NLp_irreps1.shape[0])
point_colors = color_map(norm(range(0,NLp_irreps1.shape[0]+1)))

points = [
    ('-',  'o'),   # 实线，红色，圆形标记
    ('--',  '^'),  # 虚线，绿色，三角形标记
    ('-.',  's'),  # 点划线，蓝色，方形标记
    (':',  'D'),   # 点线，青色，菱形标记
    ('-',  '*'),   # 实线，品红色，星形标记
    ('--',  'x'),  # 虚线，黄色，x形标记
    ('-.',  '+'),  # 点划线，黑色，加号标记
    (':',  'v'),   # 点线，红色，倒三角形标记
    ('-',  '<'),   # 实线，绿色，左三角形标记
    ('--',  '>'),  # 虚线，蓝色，右三角形标记
    ('-.',  '1'),  # 点划线，青色，一角星标记
    (':',  '2'),   # 点线，品红色，二角星标记
    ('-',  '3'),   # 实线，黄色，三角星标记
    ('--',  '4'),  # 虚线，黑色，四角星标记
    ('-.', 'p'),  # 点划线，红色，五边形标记
    (':',  'h'),   # 点线，绿色，六边形标记1
    ('-', 'H'),   # 实线，蓝色，六边形标记2
    ('--',  '|'),  # 虚线，青色，垂直线标记
    ('-.',  '_'),  # 点划线，品红色，水平线标记
    (':',  ','),   # 点线，黄色，像素标记
]

linestyles = [tmp1 for tmp1, tmp2 in points]
markers = [tmp2 for tmp1, tmp2 in points]

# fig, axs = plt.subplots(figsize=(12, 8))
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
for ii, _ in enumerate(NLp_irreps1):
    # axes[ii//2,np.mod(ii,2)].plot(inc_omega, NLp_irreps1[ii],  label=r"%s - |m|=%d" % (label1, ii), color=point_colors[ii], linestyle=linestyles[0], marker=markers[0], markersize=3)
    # axes[ii//2,np.mod(ii,2)].plot(inc_omega, NLp_irreps2[ii], label=r"%s    - |m|=%d" % (label2, ii), color=point_colors[ii], linestyle=linestyles[2], marker=markers[1], markersize=3)
    axes[ii//2,np.mod(ii,2)].plot(inc_omega, NLp_irreps1[ii],  label=r"%s - |m|=%d" % (label1, ii), linestyle=linestyles[0], color=point_colors[0], marker=markers[0], markersize=3)
    axes[ii//2,np.mod(ii,2)].plot(inc_omega, NLp_irreps2[ii], label=r"%s    - |m|=%d" % (label2, ii), linestyle=linestyles[1], color=point_colors[1], marker=markers[1], markersize=3)


for ax in axes.flat:
    ax.set(xlabel="$\omega$", ylabel=r"$T(\omega)$")
    ax.legend(loc="best")

# 只在外边缘显示 x 轴和 y 轴标签，去除多余的标签
# for ax in axes.flat:
#     ax.label_outer()

# plt.xlim(left=0.0)
# plt.ylim(bottom=0.0)
# plt.legend(loc="best")
plt.tight_layout()
# plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
# plt.xlabel("$\omega$")
# plt.ylabel(r"$T(\omega)$")

# plt.tick_params(labelsize=14)
plt.savefig(path_savefig, dpi=600)
plt.show()
