#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import copy
import os
import sys

import numpy as np
import ovito.vis
import phonopy as ph
from ovito import scene
from ovito.io import export_file, import_file
from ovito.modifiers import ReplicateModifier, UnwrapTrajectoriesModifier
from ovito.vis import CoordinateTripodOverlay, Viewport
from PySide6 import QtCore

# from ovito.plugins.ParticlesPython import POSCARImporter


def animate_mode(
    dim="1 1 1",
    q=[0, 0, 0],
    md=0,
    n=100,
    amp=20,
    symprec=1e-5,
    fname="force_constants.hdf5",
    rep_supercell=None,
    movname=None,
    keep_aposcars=False,
    view_type="top",
    res=[800, 600],
    phpy_obj=None,
    tripod=True,
    oscene=None,
    oid=0,
):

    if phpy_obj is None:
        scstr = dim.split()
        if len(scstr) == 3:
            supercell = [
                [float(scstr[0]), 0.0, 0.0],
                [0.0, float(scstr[1]), 0.0],
                [0.0, 0.0, float(scstr[2])],
            ]
        elif len(scstr) == 9:
            supercell = [
                [float(scstr[0]), float(scstr[1]), float(scstr[2])],
                [float(scstr[3]), float(scstr[4]), float(scstr[5])],
                [float(scstr[6]), float(scstr[7]), float(scstr[8])],
            ]
        else:
            print("invalid super cell definition")
            sys.exit()
        supercell = np.asarray(supercell)

        unitcell = ph.interface.vasp.read_vasp("POSCAR")
        # geo = geometry.geo("POSCAR")

        # funnily enough the list part is mandatory
        # unitcell = ph.structure.atoms.PhonopyAtoms(symbols=list(geo.elems), cell=geo.cell, positions=geo.coors)

        phonon = ph.Phonopy(unitcell, supercell, symprec=symprec)

        # currently using default file name
        if fname.split(".")[-1] == "hdf5":
            fc = ph.file_IO.read_force_constants_hdf5(filename=fname)
        else:
            fc = ph.file_IO.parse_FORCE_CONSTANTS(filename=fname)

        phonon.set_force_constants(fc)
    else:
        phonon = phpy_obj

    # phonon.set_dynamical_matrix()

    phonon.write_animation(
        q,
        anime_type="poscar",
        band_index=md,
        amplitude=amp,
        num_div=n,
        shift=None,
        filename=None,
    )

    if oscene is not None:
        scene.load(oscene)
        pipeline = scene.pipelines[0]
        # pipeline.add_to_scene()
        # alist = []
        # for i in range(n):
        # alist.append("APOSCAR-%03d" % (i))
        # print(POSCARImporter().import_file_sequence(alist))
        # pipeline2 = POSCARImporter().import_file_sequence(alist)
        # pipeline.source = pipeline2.source
        # print(pv.scaling.value)
        data = pipeline.compute()
        # pvscale = ovito.vis.ParticlesVis.scaling
        pipeline.source.load("APOSCAR-*")
        # pipeline.source.data.particles.vis = vis_element
        # print(pv.scaling.value)
    else:
        pipeline = import_file("APOSCAR-*", input_format="vasp")
        pipeline.modifiers.append(UnwrapTrajectoriesModifier())
        if rep_supercell is not None:
            pipeline.modifiers.append(
                ReplicateModifier(
                    adjust_box=False,
                    num_x=rep_supercell[0],
                    num_y=rep_supercell[1],
                    num_z=rep_supercell[2],
                )
            )
        pipeline.modifiers.append(UnwrapTrajectoriesModifier())
        pipeline.add_to_scene()

    data = pipeline.compute()

    if oscene is not None:
        vp = scene.viewports[oid]
        vis_element = pipeline.source.data.particles.vis
        vis_element.scaling = 0.6
    else:
        if view_type == "top":
            vtype = Viewport.Type.Top
        elif view_type == "bottom":
            vtype = Viewport.Type.Bottom
        elif view_type == "front":
            vtype = Viewport.Type.Front
        elif view_type == "back":
            vtype = Viewport.Type.Back
        elif view_type == "left":
            vtype = Viewport.Type.Left
        elif view_type == "right":
            vtype = Viewport.Type.Right

        vp = Viewport(type=vtype)

        if tripod:
            tp = CoordinateTripodOverlay()
            tp.size = 0.1
            tp.alignment = QtCore.Qt.AlignRight ^ QtCore.Qt.AlignBottom
            tp.style = CoordinateTripodOverlay.Style.Flat
            vp.overlays.append(tp)

        vp.zoom_all()
    # vp.type = Viewport.Type.Ortho
    # vp.camera_up = (1,1,1)

    # newtup = []
    # for eid,el in  enumerate(vp.camera_pos):
    # newtup.append(list(vp.camera_pos)[eid]/2)

    # vp.camera_pos = tuple(newtup)

    # vp.camera_dir = (1, 0, 0)
    # vp.fov = 60.0/60*np.pi

    # vp.render_image(size=(800,600), filename="figure.png", background=(0,0,0), frame=8)
    if movname is None:
        movname = f"animation_{md:03d}.mp4"
    vp.render_anim(size=(res[0], res[1]), filename=movname, fps=20)

    pipeline.remove_from_scene()

    if not keep_aposcars:
        os.system("rm APOSCAR-*")

    return pipeline, vp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="interface with phonopy to create animation movies with ovito"
    )

    parser.add_argument(
        "-d", dest="dim", type=str, default="1 1 1", help="supercell dimension"
    )
    parser.add_argument(
        "-q",
        dest="q",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="qpoint in reduced reciprocal coordinates",
    )
    parser.add_argument(
        "-sc",
        dest="rep_supercell",
        type=int,
        nargs=3,
        default=None,
        help="apply this supercell for visualization",
    )
    parser.add_argument(
        "-res",
        dest="res",
        type=int,
        nargs=2,
        default=[800, 600],
        help="resolution for animation",
    )
    parser.add_argument(
        "-md", dest="md", type=int, default=1, help="mode index sorted by frequency"
    )
    parser.add_argument(
        "-oid", dest="oid", type=int, default=0, help="ovito scene index"
    )
    parser.add_argument(
        "-n", dest="n", type=int, default=100, help="number of modulated files"
    )
    parser.add_argument(
        "-amp", dest="amp", type=float, default=20, help="amplitude of animation"
    )
    parser.add_argument(
        "-symprec", dest="symprec", type=float, default=1e-5, help="symmetry tolerance"
    )
    parser.add_argument(
        "-keep_aposcars",
        dest="keep_aposcars",
        action="store_true",
        default=False,
        help="do not delete the APOSCAR files after animations",
    )
    parser.add_argument(
        "-tripod",
        dest="tripod",
        action="store_true",
        default=False,
        help="draw the coordinate tripod",
    )
    parser.add_argument(
        "-fname",
        dest="fname",
        type=str,
        default="force_constants.hdf5",
        help="file name of the force constants file",
    )
    parser.add_argument(
        "-movname",
        dest="movname",
        type=str,
        default=None,
        help="name of the animation file, specify the type with the file extension",
    )
    parser.add_argument(
        "-view_type",
        dest="view_type",
        type=str,
        default="top",
        help="default view type",
    )
    parser.add_argument(
        "-scene", dest="oscene", type=str, default=None, help="scene to load with ovito"
    )

    args = parser.parse_args()

    afp = open("p_cmnd.log", "a")
    afp.write(" ".join(sys.argv) + "\n")
    afp.close()

    animate_mode(**vars(args))
