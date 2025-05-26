import os.path
import shutil



# path_save = "datas/carbon_nanotube/5x0-10x0-u1-3-defect-C5v-2"
# path_0 = "../phonon_transmission/structures/carbon_nanotube/5x0-10x0-u1-3-defect-C5v"
# path_1 = "../phonon_transmission/structures/carbon_nanotube/5x0-10x0-u1_supercell"
# idx = 6
# path_save = "datas/WS2-MoS2_NNFF-epoch2000-1/10x0-20x0-u1-3-defect-MoW-C1/10x0-20x0-u1-3-defect-MoW-C1-%d" % idx
# path_0 = "../phonon_transmission/structures/WS2-MoS2_NNFF-epoch2000-1/10x0-20x0-u1-3-defect-MoW-C1"
# path_1 = "../phonon_transmission/structures/WS2-MoS2_NNFF-epoch2000-1/10x0-20x0-WS2-MoS2-u1_supercellx1"
# idx = 1
# path_save = "datas/WS2/20x0-MoS2-u1-3-S-C1-%d" % idx
# path_0 = "../phonon_transmission/structures/MoS2/20x0-MoS2-u1-3-S-C1"
# path_1 = "../phonon_transmission/structures/MoS2/20x0-MoS2-u1_supercellx1"
idx = 1
path_save = "datas/WS2-MoS2_NNFF-epoch2000-1/20x0-MoS2-u1-3-WMo-C10v-%d" % idx
path_0 = "../phonon_transmission/structures/WS2-MoS2_NNFF-epoch2000-1/20x0-MoS2-u1-3-WMo-C10v/"
path_1 = "../phonon_transmission/structures/WS2-MoS2_NNFF-epoch2000-1/20x0-MoS2-u1_supercellx1/"
# idx = 2
# path_save = "datas/WS2-MoS2_NNFF-epoch2000-1/6x6-12x12-u1-3-WMo-C1-%d" % idx
# path_0 = "../phonon_transmission/structures/WS2-MoS2_NNFF-epoch2000-1/6x6-12x12-u1-3-WMo-C1"
# path_1 = "../phonon_transmission/structures/WS2-MoS2_NNFF-epoch2000-1/6x6-12x12_supercellx1"


path_defect_indices_source = os.path.join(path_0, "defect_indices-%d.npz" % idx)
path_defect_indices_des = os.path.join(path_save, "defect_indices.npz")
path_fc_defect_source = os.path.join(path_0, "phonon-%d_1x1x1/FORCE_CONSTANTS.continuum" % idx)
path_fc_defect_des = os.path.join(path_save, "FORCE_CONSTANTS_defect.continuum")
path_yaml_defect_source = os.path.join(path_0, "phonon-%d_1x1x1/phonopy.yaml" % idx)
path_yaml_defect_des = os.path.join(path_save, "phonopy_defect.yaml")
path_fc_block_defect_source = os.path.join(path_0, "phonon-%d_1x1x1/scatter_fc.npz" % idx)
path_fc_block_defect_des = os.path.join(path_save, "scatter_fc.npz")

path_yaml_pure_source = os.path.join(path_1, "phonon_1x1x6/phonopy.yaml")
path_yaml_pure_des = os.path.join(path_save, "phonopy_pure.yaml")
path_fc_pure_source = os.path.join(path_1, "phonon_1x1x6/FORCE_CONSTANTS.continuum")
path_fc_pure_des = os.path.join(path_save, "FORCE_CONSTANTS_pure.continuum")
path_fc_block_pure_source = os.path.join(path_1, "phonon_1x1x6/pure_fc.npz")
path_fc_block_pure_des = os.path.join(path_save, "pure_fc.npz")
path_poscar_source = os.path.join(path_1, "POSCAR")
path_poscar_des = os.path.join(path_save, "POSCAR")

if not os.path.exists(path_save):
    os.makedirs(path_save)

shutil.copy(path_defect_indices_source, path_defect_indices_des)
shutil.copy(path_fc_defect_source, path_fc_defect_des)
shutil.copy(path_yaml_defect_source, path_yaml_defect_des)
shutil.copy(path_fc_block_defect_source, path_fc_block_defect_des)

shutil.copy(path_yaml_pure_source, path_yaml_pure_des)
shutil.copy(path_fc_pure_source, path_fc_pure_des)
shutil.copy(path_fc_block_pure_source, path_fc_block_pure_des)
shutil.copy(path_poscar_source, path_poscar_des)

