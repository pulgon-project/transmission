import os.path
import shutil


path_save = "datas/WS2/6-6-u1-3-defect-3-nonTripleSym"
path_0 = "../phonon_transmission/structures/WS2/6x6-u1-3-defect-S-3-nonTripleSym"
path_1 = "../phonon_transmission/structures/WS2/6x6-u1_supercellX1"

path_defect_indices_source = os.path.join(path_0, "defect_indices.npz")
path_defect_indices_des = os.path.join(path_save, "defect_indices.npz")
path_fc_defect_source = os.path.join(path_0, "phonon_1x1x1/FORCE_CONSTANTS.continuum")
path_fc_defect_des = os.path.join(path_save, "FORCE_CONSTANTS_defect.continuum")
path_yaml_defect_source = os.path.join(path_0, "phonon_1x1x1/phonopy.yaml")
path_yaml_defect_des = os.path.join(path_save, "phonopy_defect.yaml")
path_fc_block_defect_source = os.path.join(path_0, "phonon_1x1x1/scatter_fc.npz")
path_fc_block_defect_des = os.path.join(path_save, "scatter_fc.npz")


path_yaml_pure_source = os.path.join(path_1, "phonon_1x1x4/phonopy.yaml")
path_yaml_pure_des = os.path.join(path_save, "phonopy_pure.yaml")
path_fc_pure_source = os.path.join(path_1, "phonon_1x1x4/FORCE_CONSTANTS.continuum")
path_fc_pure_des = os.path.join(path_save, "FORCE_CONSTANTS_pure.continuum")
path_fc_block_pure_source = os.path.join(path_1, "phonon_1x1x4/pure_fc.npz")
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


