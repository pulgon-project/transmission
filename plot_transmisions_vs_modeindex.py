import os.path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from ipdb import set_trace
import os
from ase.io.vasp import read_vasp


def count_folders_starting_with(directory, prefix):
    count = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            count += 1
    return count

path_0 = "datas/WS2-MoS2"
# path_0 = "datas/carbon_nanotube"
# symmetry = ["C10v-constrainSym", "10-random"]

ind1 = 2
ind2 = 2
sym1 = "C11v"
sym2 = "C1"
# sym1 = "C11v"
# sym2 = "random"
chariality = "11x0-22x0"
num_cell = 3

# path_save = os.path.join(path_0, "caroli_NL+_C10v-constrainSym-%d_random-%d" % (ind1, ind2))
path_save = os.path.join(path_0, "caroli_NL+_%s-%d_%s-%d" % (sym1, ind1, sym2, ind2))


path_poscar1 = "../phonon_transmission/structures/WS2-MoS2/%s-u1-%d-defect-S-%s/POSCAR-defect-pri-%d" % (chariality,num_cell,sym1, ind1)
path_poscar2 = "../phonon_transmission/structures/WS2-MoS2/%s-u1-%d-defect-S-%s/POSCAR-defect-pri-%d" % (chariality,num_cell,sym2, ind2)
path_poscar_pure1 = "../phonon_transmission/structures/WS2-MoS2/%s-u1-%d-defect-S-%s/POSCAR" % (chariality,num_cell,sym1)
path_poscar_pure2 = "../phonon_transmission/structures/WS2-MoS2/%s-u1-%d-defect-S-%s/POSCAR" % (chariality,num_cell,sym2)

# path_poscar1 = "../phonon_transmission/structures/carbon_nanotube/5x0-10x0-u1-3-defect-%s/POSCAR-defect-pri-%d" % (sym1, ind1)
# path_poscar_pure1 = "../phonon_transmission/structures/carbon_nanotube/5x0-10x0-u1-3-defect-%s/POSCAR" % sym1
# path_poscar2 = "../phonon_transmission/structures/carbon_nanotube/5x0-10x0-u1-3-defect-%s/POSCAR-defect-pri-%d" % (sym2, ind2)
# path_poscar_pure2 = "../phonon_transmission/structures/carbon_nanotube/5x0-10x0-u1-3-defect-%s/POSCAR" % sym2

# path_1 = os.path.join(path_0, '10x0-20x0-u1-5-defect-S-%s-constrainSym-%d' % (sym1,ind1), "transmission_irreps.npz")
# path_2 = os.path.join(path_0, '10x0-20x0-u1-5-defect-S-10-%s-%d' % (sym2,ind2), "transmission_irreps.npz")
path_1 = os.path.join(path_0, '%s-u1-%d-defect-S-%s-%d' % (chariality,num_cell,sym1,ind1), "transmission_irreps.npz")
path_2 = os.path.join(path_0, '%s-u1-%d-defect-S-%s-%d' % (chariality,num_cell,sym2,ind2), "transmission_irreps.npz")


colors = [value for key, value in mcolors.XKCD_COLORS.items()]
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

ind = 0
fig, axs = plt.subplots(figsize=(12, 8))


pure_poscar = read_vasp(path_poscar_pure1)

poscar1 = read_vasp(path_poscar1)
poscar2 = read_vasp(path_poscar2)

# num_defect1 = len(pure_poscar) - len(np.where(poscar1.numbers==6)[0])
# num_defect2 = len(pure_poscar) - len(np.where(poscar1.numbers==6)[0])
num_defect1 = len(np.where(pure_poscar.numbers==16)[0]) - len(np.where(poscar1.numbers==16)[0])
num_defect2 = len(np.where(pure_poscar.numbers==16)[0]) - len(np.where(poscar1.numbers==16)[0])

data1 = np.load(path_1)
data2 = np.load(path_2)
trans1 = data1["trans"]
trans2 = data2["trans"]
inc_omega = data1["inc_omega"]
NLp = data1["NLp"]


plt.plot(inc_omega, NLp, label=r"$Pure-N_{L+}$", color="grey")
plt.plot(np.array(inc_omega), trans1, label=r"Caroli_%s-%d-%d S" % (sym1, ind1, num_defect1), color=colors[0], linestyle=linestyles[0], marker=markers[0], markersize=3)
plt.plot(np.array(inc_omega), trans2, label=r"Caroli_C1-%d-%d S" % (ind2, num_defect2), color=colors[1], linestyle=linestyles[1], marker=markers[1], markersize=3)

plt.xlim(left=0.0)
plt.ylim(bottom=0.0)
plt.legend(loc="best")
# plt.tight_layout()
plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
plt.ylabel(r"$T(\omega)$")
# plt.tick_params(labelsize=14)
# plt.savefig(path_save, dpi=600)
plt.show()
