#!/usr/bin/env python
# Copyright 2018 Jesús Carrete Montaña <jesus.carrete.montana@tuwien.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import glob
import sys
import os
import os.path
import math
import copy
import math
import argparse
import itertools
import collections
import functools
import pickle

import gzip
import lzma
import tqdm
import numpy as np
import numpy.linalg as nla
import scipy as sp
import scipy.linalg as la
import scipy.spatial
import scipy.spatial.distance
import ase
import ase.io
import ase.data
import matplotlib
import matplotlib.pyplot as plt

import decimation
import translation_rules

matplotlib.rcParams["font.size"] = 16.

NPOINTS = 201


# TODO: Add explicit symmetrization for when we decide to use DFT constants.
def copen(filename, mode):
    """Wrapper around open() that also handles gz and xz files."""
    ext = os.path.splitext(filename)[1]
    if ext == ".gz":
        return gzip.open(filename, mode)
    elif ext == ".xz":
        return lzma.open(filename, mode)
    else:
        return open(filename, mode)


def next_nonblank(f):
    """Return the next non-blank line from a file-like object."""
    while True:
        l = next(f)
        if len(l.strip()) != 0:
            return l


def read_lammps(filename):
    """Read one of the LAMMPS files provided by Riccardo Rurali and return
    an Atoms object.
    """
    with copen(filename, "r") as f:
        next_nonblank(f)
        natoms = int(next_nonblank(f).split()[0])
        ntypes = int(next_nonblank(f).split()[0])
        limits = []
        for i in range(3):
            fields = next_nonblank(f).split()
            m = float(fields[0])
            M = float(fields[1])
            if not np.isclose(m, 0.):
                raise ValueError("The cell must have its origin at 0")
            limits.append(M)
        if min(limits) < 0.:
            raise ValueError("An axis has the wrong orientation")
        xy, xz, yz = [float(i) for i in next(f).strip().split()[:3]]
        if not np.allclose([xz, yz], 0.):
            raise ValueError("Only xy tilts are supported")
        cell = np.array([[limits[0], 0., 0.], [xy, limits[1], 0.],
                         [xz, yz, limits[2]]])
        next_nonblank(f)
        elements = []
        for i in range(ntypes):
            fields = next_nonblank(f).split()
            mass = float(fields[1])
            Z = np.abs(ase.data.atomic_masses - mass).argmin()
            elements.append(ase.data.chemical_symbols[Z])
        next_nonblank(f)
        types = []
        xyz = []
        for i in range(natoms):
            fields = next_nonblank(f).split()
            types.append(int(fields[1]) - 1)
            xyz.append([float(j) for j in fields[2:5]])
        xyz = np.array(xyz)
    elements = [elements[i] for i in types]
    nruter = ase.Atoms(symbols=elements, cell=cell, positions=xyz)
    return nruter


def ssqrt(x):
    """'Signed' square root of real numbers."""
    return np.sign(x) * np.sqrt(np.abs(x))


def get_direct_coordinates(atoms):
    """Obtain atoms.get_scaled_positions() and reduce everything to the
    interval [0., 1.).
    """
    nruter = atoms.get_scaled_positions() % 1.
    nruter[np.isclose(nruter, 1.)] = 0.
    return nruter


def circular_mean(x):
    """Circular mean of a vector of quantities in the [0, 1) interval."""
    if x.min() < 0. or x.max() >= 1.:
        raise ValueError("All values must fall in the [0, 1) interval")
    return np.arctan2(
        np.sin(2. * np.pi * x).sum(),
        np.cos(2. * np.pi * x).sum()) / 2. / np.pi


def find_layers(atoms, *args, **kwargs):
    """Return a list of lists of indices of atoms belonging to the first
    element in the structure and sharing a z coordinate.
    """
    elements = atoms.get_chemical_symbols()
    e0 = elements[0]
    indices = [i for i, e in enumerate(elements) if e == e0]
    zvec = get_direct_coordinates(atoms)[indices, 2]
    nruter = []
    for iz, z in enumerate(zvec):
        for c in nruter:
            delta = z - zvec[c[0]]
            delta -= round(delta)
            if np.isclose(delta, 0., *args, **kwargs):
                c.append(iz)
                break
        else:
            nruter.append([iz])
    nruter = [[indices[j] for j in i] for i in nruter]
    return nruter


def find_all_layers(atoms, *args, **kwargs):
    """Return a dictionary of list of lists of indices of atoms belonging to
    the same chemical element and sharing a z coordinate. Each key is the
    element in question.
    """
    elements = atoms.get_chemical_symbols()
    ekeys = list(dict.fromkeys(elements))
    nruter = dict()
    for e0 in ekeys:
        indices = [i for i, e in enumerate(elements) if e == e0]
        zvec = get_direct_coordinates(atoms)[indices, 2]
        nruter[e0] = []
        for iz, z in enumerate(zvec):
            for c in nruter[e0]:
                delta = z - zvec[c[0]]
                delta -= round(delta)
                if np.isclose(delta, 0., *args, **kwargs):
                    c.append(iz)
                    break
            else:
                nruter[e0].append([iz])
        nruter[e0] = [[indices[j] for j in i] for i in nruter[e0]]
    return nruter


def are_close(a, b, *args, **kwargs):
    """Return true if two reduced coordinate vectors are compatible."""
    delta = a - b
    delta -= np.round(delta)
    return np.isclose(la.norm(delta), 0., *args, **kwargs)


def next_letter(l):
    """Return the next capital letter in the alphabet. The result is undefined
    if l is not a single letter from A to Y.
    """
    return chr(ord(l) + 1)


def label_layers(atoms, layers, atol, start="A"):
    """Return a set of alphabetic labels for a set of layers obtained using
    find_layers().
    """
    nlayers = len(layers)
    if nlayers == 0:
        return ""
    coords = get_direct_coordinates(atoms)
    nruter = start
    representatives = {nruter: coords[layers[0], :2]}

    for l in layers[1:]:
        for r in representatives:
            candidates = copy.copy(representatives[r]).tolist()
            for i in coords[l, :2]:
                for c in candidates:
                    if are_close(i, c, atol=atol):
                        candidates.remove(c)
                        break
                else:
                    break
            else:
                nruter += r
                break
        else:
            letter = next_letter(max(nruter))
            nruter += letter
            representatives[letter] = coords[l, :2]
    return nruter


def complete_layers(atoms, layers):
    """Complete the provided set of layer indices with the remaining atoms."""
    # The Atoms object must contain an integer number of copies of the motif.
    # If the motif contains n atoms, find the n - 1 atoms closer to the first
    # atom in the first layer that are not included in any layer, and then
    # look for all other copies of the motif thus built.
    nruter = copy.deepcopy(layers)
    coords = get_direct_coordinates(atoms)
    natoms = coords.shape[0]
    found = set(sum(nruter, []))
    nfound = len(found)
    if natoms % nfound:
        raise ValueError(
            "The Atoms object does not contain an integer number of layers")
    nmotif = natoms // nfound
    motif0 = [layers[0][0]]
    pos0 = coords[layers[0][0], :]
    # Since we want to keep pos0 at the min-z boundary of the motif, in looking
    # for distances we always take images of each atom that lie towards higher
    # values of z.
    notfound = set(range(natoms)) - found
    notfound = sorted(list(notfound))
    for im in range(1, nmotif):
        distances = np.empty(len(notfound))
        for iia, ia in enumerate(notfound):
            pos = coords[ia, :]
            delta = pos - pos0
            delta -= np.round(delta)
            if delta[2] < 0.:
                delta[2] += 1.
            cdelta = atoms.get_cell().T @ delta
            distances[iia] = la.norm(cdelta)
        index = distances.argmin()
        nextatom = notfound[index]
        motif0.append(nextatom)
        found.add(nextatom)
        del notfound[index]
    # But once the motif has been identified, any copy of the vectors joining
    # atoms in the motif will do.
    found = set(sum(nruter, []))
    notfound = set(range(natoms)) - found
    notfound = sorted(list(notfound))
    for ia in motif0[1:]:
        delta = coords[ia, :] - pos0
        for o, n in zip(layers, nruter):
            for i in o:
                target = coords[i, :] + delta
                diffs = np.empty(len(notfound))
                for ik, k in enumerate(notfound):
                    diff = coords[k, :] - target
                    diff -= np.round(diff)
                    diffs[ik] = la.norm(diff)
                index = diffs.argmin()
                n.append(notfound[index])
                del notfound[index]
    # Sanity checks.
    found = sorted(sum(nruter, []))
    if found != list(range(natoms)):
        raise ValueError("Invalid layer structure")
    natoms = len(nruter[0])
    for i in nruter[1:]:
        if len(i) != natoms:
            raise ValueError("Invalid layer structure")
    return nruter


def read_FORCE_CONSTANTS(filename):
    """Read a Phonopy FORCE_CONSTANTS file compressed with xz. Fused-together
    atom indices are ignored.
    """
    with copen(filename, mode="r") as f:
        natoms = int(next(f).split()[0])
        if natoms > 9999:
            raise ValueError(
                "Files with more than 9999 atoms are not supported")
        nruter = np.empty((natoms, natoms, 3, 3))
        for i in range(natoms):
            for j in range(natoms):
                next(f)
                for k in range(3):
                    nruter[i, j, k, :] = [float(m) for m in next(f).split()]
    return nruter


def unfold_fcs(fcs, atoms, ucells):
    """Split the FCs into blocks joining different unit cells. Split blocks
    when necessary in order to restore the symmetry. This unfolding is only
    performed along the first two axes.
    """
    natoms = len(ucells[list(ucells.keys())[0]])
    nx = max(i[0] for i in ucells) + 1
    ny = max(i[1] for i in ucells) + 1
    nz = max(i[2] for i in ucells) + 1
    coords = get_direct_coordinates(atoms).copy()
    deltax = find_best_origin(coords[:, 0])
    deltay = find_best_origin(coords[:, 1])
    coords[:, 0] = (coords[:, 0] - deltax) % 1.
    coords[:, 1] = (coords[:, 1] - deltay) % 1.
    # Precalculate all distances to avoid expensive Python loops later on.
    disps = []
    disps = list(itertools.product(range(-1, 2), range(-1, 2)))
    cartesian = coords @ atoms.get_cell()
    d2s = np.empty((9, cartesian.shape[0], cartesian.shape[0]))
    for s, (sx, sy) in enumerate(disps):
        cartesianp = (coords + [sx, sy, 0.]) @ atoms.get_cell()
        d2s[s, :, :] = sp.spatial.distance.cdist(cartesian, cartesianp,
                                                 "sqeuclidean")
    # There are plenty of equivalent pairs of atoms in the simulation box.
    # Compile a dictionary of nonequivalent 3x3 subblocks of force constants
    # from which to build the final matrices.
    pieces = dict()
    allpieces = collections.defaultdict(list)
    deltas = collections.defaultdict(list)
    for i in tqdm.tqdm(ucells, desc="1st unit cell"):
        l1 = i[2]
        for j in ucells:
            l2 = j[2]
            dx = j[0] - i[0]
            dy = j[1] - i[1]
            subd2s = d2s[:, ucells[i], :][:, :, ucells[j]]
            d2min = subd2s.min(axis=0)
            degenerate = np.isclose(subd2s, d2min)
            nequiv = degenerate.sum(axis=0)
            for a1, a2 in itertools.product(range(natoms), range(natoms)):
                for s in np.nonzero(degenerate[:, a1, a1])[0]:
                    sx, sy = disps[s]
                    pieces[(l1, l2, dx + sx * nx, dy + sy * ny, a1,
                            a2)] = (fcs[ucells[i][a1], ucells[j][a2], :, :] /
                                    nequiv[a1, a2])
                    allpieces[(l1, l2, dx + sx * nx, dy + sy * ny, a1,
                               a2)].append(
                                   (fcs[ucells[i][a1], ucells[j][a2], :, :] /
                                    nequiv[a1, a2]))
                    deltas[(l1, l2, dx + sx * nx, dy + sy * ny, a1,
                            a2)].append((i, j))

    nruter = collections.defaultdict(
        functools.partial(np.zeros, (3 * natoms, 3 * natoms)))
    for i in pieces:
        l1, l2, dx, dy, a1, a2 = i
        nruter[(l1, l2, dx,
                dy)][3 * a1:3 * (a1 + 1), 3 * a2:3 * (a2 + 1)] = pieces[i]
    return nruter


def calc_centers(atoms, layers):
    """Return the average reduced Z coordinate of each layer."""
    coords = get_direct_coordinates(atoms)[:, 2]
    return np.array([coords[l].mean() for l in layers])


def sort_layers(atoms, layers, labels):
    """Sort a set of layers and their labels according to the average position
    along the OZ axis and return the result.
    """
    centers = calc_centers(atoms, layers)
    order = centers.argsort()
    return ([layers[i] for i in order], "".join(labels[i] for i in order))


def calc_distances(atoms, layers):
    """Return a dictionary of normal distances between layers."""
    nlayers = len(layers)
    centers = calc_centers(atoms, layers)
    nruter = dict()
    for i in range(nlayers):
        for j in range(nlayers):
            delta = centers[i] - centers[j]
            delta -= round(delta)
            nruter[(i, j)] = abs(delta) * atoms.get_cell()[2, 2]
    return nruter


def find_label_motif(labels):
    """Return the shortest contiguous motif that can recreate a label
    sequence.
    """
    nlabels = len(labels)
    # Generate a list of divisors as possible lengths of the motif.
    divisors = []
    for i in range(1, int(math.sqrt(nlabels) + 1)):
        if nlabels % i == 0:
            divisors.append(i)
            if i * i != nlabels:
                divisors.append(nlabels // i)
    divisors.sort()
    divisors = divisors[:-1]
    for i in divisors:
        motif = labels[:i]
        reconstructed = motif * (nlabels // i)
        if reconstructed == labels:
            return motif
    return labels


def find_displacement(atoms, a, b, *args, **kwargs):
    """Try to find a rigid displacement along a single axis connecting the
    subset of atoms with indices in a to the set of atoms with indices in b.
    Return True or False depending on whether such displacement exists.
    """
    natoms = len(a)
    if len(b) != natoms:
        return None
    coords = get_direct_coordinates(atoms)
    els = atoms.get_chemical_symbols()
    coordsa = np.copy(coords[a, :])
    elsa = [els[i] for i in a]
    coordsb = np.copy(coords[b, :])
    elsb = [els[i] for i in b]

    # Try all possible displacements connecting atom 0 of set a with any atom
    # of the same element in set b and involving a displacement along only
    # one axis.
    distances = np.empty(natoms)
    for i in range(natoms):
        if elsb[i] == elsa[0]:
            delta = coordsb[i, :] - coordsa[0, :]
            delta -= np.round(delta)
            if np.count_nonzero(np.isclose(delta, 0., *args, **kwargs)) < 2:
                distances[i] = np.infty
            else:
                delta = atoms.get_cell().T @ delta
                distances[i] = la.norm(delta)
        else:
            distances[i] = np.infty
    candidates = distances.argsort()
    distances = distances[candidates]
    candidates = candidates[np.isfinite(distances)]

    for candidate in candidates:
        displacement = coordsb[candidate, :] - coordsa[0, :]
        nruter = []
        for i in range(natoms):
            posi = coordsa[i, :] + displacement
            eli = elsa[i]
            for j in range(natoms):
                if j in nruter or elsb[j] != eli:
                    continue
                posj = coordsb[j, :]
                if are_close(posi, posj, *args, **kwargs):
                    nruter.append(j)
                    break
            else:
                break
        else:
            return True
    return False


def find_permutation(atoms, a, b):
    """Among all possible bijections between atoms in sets a and b that preserve
    the elements, select the one that minimizes the mean-square difference of
    positions after the geometric centers have been moved to match. Return the
    result as a permutation. Return None if there is no such permutation.

    Some elements are considered equivalents for the purposes of this function.
    """
    equivalences = {
        "Si": ("Si", "Ge"),
        "Ge": ("Ge", "Si"),
        "As": ("As", ),
        "P": ("P", ),
        "N": ("N", ),
        "Ga": ("Ga", "Al", "In"),
        "Al": ("Al", "In", "Ga"),
        "In": ("In", "Ga", "Al")
    }
    natoms = len(a)
    if len(b) != natoms:
        raise ValueError("a and b must have the same length")
    coords = get_direct_coordinates(atoms)
    els = atoms.get_chemical_symbols()
    coordsa = np.copy(coords[a, :])
    elsa = [els[i] for i in a]
    coordsb = np.copy(coords[b, :])
    elsb = [els[i] for i in b]
    centera = np.array([
        circular_mean(coordsa[:, 0]),
        circular_mean(coordsa[:, 1]),
        circular_mean(coordsa[:, 2])
    ])
    centerb = np.array([
        circular_mean(coordsb[:, 0]),
        circular_mean(coordsb[:, 1]),
        circular_mean(coordsb[:, 2])
    ])
    d2 = np.empty((natoms, natoms))
    for i in range(natoms):
        posi = coordsa[i, :] - centera
        for j in range(natoms):
            posj = coordsb[j, :] - centerb
            delta = posj - posi
            delta -= np.round(delta)
            d2[i, j] = la.norm(delta, 1)
    mintot = np.inf
    nruter = None
    for p in itertools.permutations(range(natoms)):
        elsp = [elsb[i] for i in p]
        for e1, e2 in zip(elsp, elsa):
            if e1 not in equivalences[e2]:
                break
        else:
            tot = sum(d2[i, p[i]] for i in p)
            if tot < mintot:
                mintot = tot
                nruter = list(p)
    return nruter


def find_unique_values(values0, *args, **kwargs):
    """Return a maximal sorted array of unique values from the input vector,
    taking periodicity (with period 1) into account.
    where two consecutive values are considered unique if are_close returns
    false.
    """
    if len(values0) == 0:
        return np.array([])
    values = np.copy(values0)
    values.sort()
    nruter = [values[0]]
    for i in values[1:]:
        for j in nruter:
            if are_close(i, j, *args, **kwargs):
                break
        else:
            nruter.append(i)
    return np.array(nruter)


def find_best_origin(values):
    """Find the point in [0, 1] with the maxmin distance to the set of values,
    interpreted taking periodic boundary conditions into account.
    """
    values = sorted((values % 1.).tolist())
    values.append(values[0] + 1.)
    deltas = np.diff(values)
    pos = deltas.argmax()
    nruter = (.5 * (values[pos] + values[pos + 1])) % 1.
    return nruter


def find_cell_along_axis(atoms, axis, atol):
    """Find the largest integer n such that the simulation box can be divided
    into n identical parts along a particular axis.
    """
    coords = get_direct_coordinates(atoms)
    natoms = coords.shape[0]
    if natoms == 0:
        raise ValueError("Empty Atoms object")
    unique = find_unique_values(coords[:, axis], atol=atol)
    delta = find_best_origin(unique)
    for i in range(len(unique), 1, -1):
        # Put each atom in a bin.
        indices = np.digitize((coords[:, axis] - delta) % 1.,
                              np.arange(i + 1.) / float(i)) - 1
        bins = []
        for j in range(i):
            bins.append(np.nonzero(indices == j)[0])
        # And try to see if all bins are equivalent.
        for j in range(1, i):
            if len(bins[j]) != len(bins[0]):
                break
        else:
            for j in range(1, i):
                if not find_displacement(atoms, bins[0], bins[j], atol=atol):
                    break
            else:
                return i
    return 1


def find_unit_cells(atoms, layers, nx, ny, atol):
    """Return a set of arrays of atom indices where each array defines a unit
    cell and is sorted in a consistent way.
    """
    # Bin the atoms in each layer.
    coords = get_direct_coordinates(atoms)
    natoms = coords.shape[0]
    bins = collections.defaultdict(list)
    deltax = find_best_origin(coords[:, 0])
    deltay = find_best_origin(coords[:, 1])
    for il, l in enumerate(layers):
        indx = np.digitize((coords[l, 0] - deltax) % 1.,
                           np.arange(nx + 1.) / float(nx)) - 1
        indy = np.digitize((coords[l, 1] - deltay) % 1.,
                           np.arange(ny + 1.) / float(ny)) - 1
        for ix, iy, ia in zip(indx, indy, l):
            bins[(ix, iy, il)].append(ia)
    # Sort the atoms in each bin.
    b0 = bins[(0, 0, 0)]
    keys = list(bins.keys())
    keys.remove((0, 0, 0))
    for k in keys:
        b = bins[k]
        perm = find_permutation(atoms, b0, b)
        bins[k] = [bins[k][i] for i in perm]
    return bins


def fold_in_plane(fcs, qa, qb, verbose=False):
    """Fourier-transform a set of force-constant blocks along the two in-plane
    directions, resulting in a new set of (potentially complex) blocks that can
    be used to build the dynamical matrix through a 1D Fourier transform.
    """
    block0 = fcs[list(fcs)[0]]
    nruter = collections.defaultdict(
        functools.partial(np.zeros_like, block0, dtype=np.complex128))
    for k in fcs:
        block = fcs[k]
        expfactor = np.exp(-1.j * (qa * k[-2] + qb * k[-1]))
        short_key = k[:-2]
        if len(short_key) == 1:
            short_key = short_key[0]
        nruter[short_key] += block * expfactor
    return dict(nruter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the force constants for a Rurali interface system"
    )
    parser.add_argument(
        "-e",
        "--eps",
        type=float,
        default=1e-5,
        help="prefactor for the imaginary part of the energies")
    parser.add_argument(
        "-g",
        "--geometric-tolerance",
        type=float,
        default=1e-4,
        help="tolerance to be used when comparing positions in space")
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=1e-4,
        help="if a mode's eigenvalue has modulus >1 - tolerance, consider"
        " it a propagating mode")
    parser.add_argument("-d",
                        "--decimation",
                        type=float,
                        default=1e-8,
                        help="tolerance for the decimation procedure")
    parser.add_argument(
        "-m",
        "--maxiter",
        type=int,
        default=100000,
        help="maximum number of iterations in the decimation loop")
    parser.add_argument("poscar_file", help="VASP/LAMMPS position file")
    parser.add_argument("fc_file", help="force constant file")
    parser.add_argument("qa",
                        help="Wave vector along the first in-plane direction")
    parser.add_argument("qb",
                        help="Wave vector along the second in-plane direction")
    parser.add_argument("odir", help="output directory")
    args = parser.parse_args()

    qa = float(args.qa)
    qb = float(args.qb)

    pickle_file = os.path.splitext(args.fc_file)[0] + ".pkl"

    try:
        atoms = ase.io.read(args.poscar_file)
    except ValueError:
        atoms = read_lammps(args.poscar_file)
    atoms.set_scaled_positions(get_direct_coordinates(atoms))
    natoms = atoms.get_positions().shape[0]

    layers = find_layers(atoms, atol=args.geometric_tolerance)
    all_layers = find_all_layers(atoms, atol=args.geometric_tolerance)
    # TODO: Generalize the logic to arbitrary combinations of elements.
    all_layers.pop("P", None)
    all_layers.pop("As", None)
    ekeys = list(all_layers.keys())
    if len(ekeys) == 1:
        layers = all_layers[ekeys[0]]
        labels = label_layers(atoms, layers, args.geometric_tolerance)
    elif len(ekeys) == 2:
        all_layers[ekeys[0]] = all_layers[ekeys[0]][::2]
        all_layers[ekeys[1]] = all_layers[ekeys[1]][::2]
        labels = label_layers(atoms, all_layers[ekeys[0]],
                              args.geometric_tolerance)
        labels += label_layers(atoms, all_layers[ekeys[1]],
                               args.geometric_tolerance,
                               next_letter(max(labels)))
        layers = all_layers[ekeys[0]] + all_layers[ekeys[1]]
    else:
        sys.exit("Error: there are more than two elements other than P/As")
    # layers, labels = sort_layers(atoms, layers, labels)
    distances = calc_distances(atoms, layers)
    nlayers = len(labels)
    print("Sequence of layers in the file:")
    print(labels)
    print("Number of layers:", nlayers)
    if nlayers % 4:
        sys.exit("Error: expecting a multiple of 4")
    nhalf = nlayers // 4
    part1 = labels[-nhalf:] + labels[:nhalf]
    part2 = labels[nhalf:-nhalf]
    aL = distances[(0, 1)]
    aR = distances[(nhalf, nhalf + 1)]
    print("aL =", aL)
    print("aR =", aR)
    motifL = find_label_motif(part1)
    motifR = find_label_motif(part2)
    print("First motif:", motifL)
    print("Second motif:", motifR)

    layers = complete_layers(atoms, layers)
    nx = find_cell_along_axis(atoms, 0, args.geometric_tolerance)
    ny = find_cell_along_axis(atoms, 1, args.geometric_tolerance)
    print(f"In-plane unit cells: {nx} x {ny}")
    nacell = natoms // nx // ny // nlayers
    print(f"Atoms per unit cell: {nacell}")
    ucells = find_unit_cells(atoms, layers, nx, ny, args.geometric_tolerance)

    pmasses = dict()
    for u in ucells:
        pmasses[u[2]] = sum([[math.sqrt(i)] * 3
                             for i in atoms.get_masses()[ucells[u]]], [])
    print("Size of a matrix block:", 3 * nacell)

    if os.path.isfile(pickle_file):
        print("Preprocessed force constant file found:", pickle_file)
        print("Reading the preprocessed interatomic force constants")
        fcs = pickle.load(open(pickle_file, "rb"))
    else:
        print("Reading the unprocessed interatomic force constants")
        fcs = read_FORCE_CONSTANTS(args.fc_file)
        if atoms.get_positions().shape[0] != fcs.shape[0]:
            sys.exit("Error: the shapes of coordinates and FCs do not match")
        print("Unfolding the IFCs:")
        fcs = unfold_fcs(fcs, atoms, ucells)
        print("Saving the preprocessed force constants to", pickle_file)
        pickle.dump(fcs, open(pickle_file, "wb"))

    print(f"{len(fcs)} IFC blocks after unfolding and removing redundancies")

    # Extract sub-dictionaries of constants for the leads.
    print("Extracting the IFC matrices for the leads")
    par_indices = set(k[2:] for k in fcs)
    startL = nlayers - len(motifL)
    startR = nlayers // 2 - len(motifR)
    fcsL = dict()
    fcsR = dict()
    for k in par_indices:
        fcsL[(0, k[0], k[1])] = np.block([[
            fcs[(startL + i, startL + j, k[0], k[1])]
            for j in range(len(motifL))
        ] for i in range(len(motifL))])
        fcsL[(1, k[0], k[1])] = np.block(
            [[fcs[(startL + i, j, k[0], k[1])] for j in range(len(motifL))]
             for i in range(len(motifL))])
        fcsL[(-1, -k[0], -k[1])] = fcsL[(1, k[0], k[1])].T
        fcsR[(0, k[0], k[1])] = np.block([[
            fcs[(startR + i, startR + j, k[0], k[1])]
            for j in range(len(motifR))
        ] for i in range(len(motifR))])
        fcsR[(1, k[0], k[1])] = np.block([[
            fcs[(startR + i, startR + len(motifR) + j, k[0], k[1])]
            for j in range(len(motifR))
        ] for i in range(len(motifR))])
        fcsR[(-1, -k[0], -k[1])] = fcsR[(1, k[0], k[1])].T
    print("\t -{} matrices for the left lead".format(len(fcsL)))
    print("\t -{} matrices for the right lead".format(len(fcsR)))

    # Symmetrize the force constants for the leads.
    symmetrizerL = translation_rules.BulkSymmetrizer(fcsL)
    fcsL = symmetrizerL.get_fixed_fcs()
    symmetrizerR = translation_rules.BulkSymmetrizer(fcsR)
    fcsR = symmetrizerR.get_fixed_fcs()

    # Build the blocks necessary for the joint symmetrization.
    TL = dict()
    HL = dict()
    VLC = dict()
    KC = dict()
    VCR = dict()
    HR = dict()
    TR = dict()
    # These will be left untouched...
    for k in fcsL:
        if k[0] == 0:
            HL[(k[1], k[2])] = fcsL[k]
            HR[(k[1], k[2])] = fcsR[k]
        elif k[0] == 1:
            TL[(k[1], k[2])] = fcsL[k]
            TR[(k[1], k[2])] = fcsR[k]
    # ...and these will be adjusted to fulfill the sum rules.
    motifC = motifL + motifR
    start = nhalf - len(motifL)
    for k in par_indices:
        KC[k] = np.block([[
            fcs[(start + i, start + j, k[0], k[1])] for j in range(len(motifC))
        ] for i in range(len(motifC))])
        # And the matrices describing the interaction with the leads.
        start1 = nhalf - 2 * len(motifL)
        start2 = nhalf - len(motifL)
        VLC[k] = np.block([[
            fcs[(start1 + i, start2 + j, k[0], k[1])]
            for j in range(len(motifC))
        ] for i in range(len(motifL))])
        start1 = nhalf - len(motifL)
        start2 = nhalf + len(motifR)
        VCR[k] = np.block([[
            fcs[(start1 + i, start2 + j, k[0], k[1])]
            for j in range(len(motifR))
        ] for i in range(len(motifC))])

    # Symmetrize the force constants of the whole system.
    symmetrizer = translation_rules.InterfaceSymmetrizer(
        TL, HL, VLC, KC, VCR, HR, TR)
    VLC, KC, VCR = symmetrizer.get_fixed_fcs()
    for k in par_indices:
        r_span, c_span = fcs[(start, start, k[0], k[1])].shape
        for i in range(len(motifC)):
            r_offset = i * r_span
            for j in range(len(motifC)):
                c_offset = j * c_span
                fcs[(start + i, start + j, k[0],
                     k[1])][:, :] = KC[k][r_offset:r_offset +
                                          r_span, c_offset:c_offset + c_span]
        # And the matrices describing the interaction with the leads.
        start1 = nhalf - 2 * len(motifL)
        start2 = nhalf - len(motifL)
        r_span, c_span = fcs[(start1, start2, k[0], k[1])].shape
        for i in range(len(motifL)):
            r_offset = i * r_span
            for j in range(len(motifC)):
                c_offset = j * c_span
                fcs[(start1 + i, start2 + j, k[0],
                     k[1])][:, :] = VLC[k][r_offset:r_offset +
                                           r_span, c_offset:c_offset + c_span]
        start1 = nhalf - len(motifL)
        start2 = nhalf + len(motifR)
        r_span, c_span = fcs[(start1, start2, k[0], k[1])].shape
        for i in range(len(motifC)):
            r_offset = i * r_span
            for j in range(len(motifR)):
                c_offset = j * c_span
                fcs[(start1 + i, start2 + j, k[0],
                     k[1])][:, :] = VCR[k][r_offset:r_offset +
                                           r_span, c_offset:c_offset + c_span]
    # Turn this into a one-dimensional problem.
    # First, fold the force constants for the leads...
    onedfcsL = fold_in_plane(fcsL, qa, qb)
    onedfcsR = fold_in_plane(fcsR, qa, qb)
    # ...and reduce them by the masses...
    massesL = pmasses[startL] * len(motifL)
    smassesL = np.outer(massesL, massesL)
    for k in onedfcsL:
        onedfcsL[k] *= (1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J /
                        smassesL)  # (rad / ps)**2
    massesR = pmasses[startR] * len(motifR)
    smassesR = np.outer(massesR, massesR)
    for k in onedfcsR:
        onedfcsR[k] *= (1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J /
                        smassesR)  # (rad / ps)**2
    # ...and then do the same for the whole structure to extract the constants
    # for the defective segemnt.
    onedfcs = fold_in_plane(fcs, qa, qb)
    for k in onedfcs:
        smasses = np.outer(pmasses[k[0]], pmasses[k[1]])
        onedfcs[k] *= (1e-24 * ase.units.m**2 * ase.units.kg / ase.units.J /
                       smasses)  # (rad / ps)**2

    if False:
        x = []
        y = []
        for k in onedfcs:
            dl = (k[0] - k[1]) % nlayers
            dl = min(abs(dl), abs(nlayers - dl))
            x.append(dl)
            y.append(la.norm(onedfcs[k]))
        x = np.array(x)
        y = np.array(y)
        order = x.argsort()
        x = x[order]
        y = y[order]
        plt.figure()
        plt.scatter(x, y)
        plt.xticks(np.arange(10))
        plt.xlim(left=-.5, right=10)
        plt.ylim(bottom=0.)
        plt.yscale("symlog", linthreshy=1e-8)
        plt.xlabel("Distance between segments")
        plt.ylabel("Norm of the FC block")
        plt.tight_layout()

    # Extract the basic blocks for both bulk leads.
    HL = onedfcsL[0]
    TL = onedfcsL[1]
    HR = onedfcsR[0]
    TR = onedfcsR[1]

    # Build the effective force constant matrix for the central region around
    # the interface.
    KC = np.block(
        [[onedfcs[(start + i, start + j)] for j in range(len(motifC))]
         for i in range(len(motifC))])
    # And the matrices describing the interaction with the leads.
    start1 = nhalf - 2 * len(motifL)
    start2 = nhalf - len(motifL)
    VLC = np.block(
        [[onedfcs[(start1 + i, start2 + j)] for j in range(len(motifC))]
         for i in range(len(motifL))])
    start1 = nhalf - len(motifL)
    start2 = nhalf + len(motifR)
    VCR = np.block(
        [[onedfcs[(start1 + i, start2 + j)] for j in range(len(motifR))]
         for i in range(len(motifC))])

    # Obtain the phonon spectra of both bulk leads.
    qvec = np.linspace(0., 2. * np.pi, num=1001)
    omegaL, vgL = decimation.q2omega(HL, TL, qvec)
    omegaR, vgR = decimation.q2omega(HR, TR, qvec)

    # Compute all the parameters of the interface
    inc_omega = np.linspace(1e-4,
                            max(omegaL.max(), omegaR.max()) * 1.01,
                            num=NPOINTS,
                            endpoint=True)
    one = np.eye(KC.shape[0], dtype=np.complex128)
    trans = np.empty_like(inc_omega)
    trans_check = np.empty_like(inc_omega)
    NLp = np.empty_like(inc_omega)
    NRp = np.empty_like(inc_omega)
    NLm = np.empty_like(inc_omega)
    NRm = np.empty_like(inc_omega)
    for iomega, omega in enumerate(inc_omega):
        en = omega * (omega + 1.j * args.eps)

        # Build the four retarded GFs of leads extending left or right.
        inv_gLretm = decimation.inv_g00(HL, TL, omega, args.eps,
                                        args.decimation, args.maxiter)
        inv_gLretp = decimation.inv_g00(HL,
                                        TL.conj().T, omega, args.eps,
                                        args.decimation, args.maxiter)
        inv_gRretm = decimation.inv_g00(HR, TR, omega, args.eps,
                                        args.decimation, args.maxiter)
        inv_gRretp = decimation.inv_g00(HR,
                                        TR.conj().T, omega, args.eps,
                                        args.decimation, args.maxiter)
        # And the four advanced versions, i.e., their Hermitian conjugates.
        inv_gLadvm = inv_gLretm.conj().T
        inv_gLadvp = inv_gLretp.conj().T
        inv_gRadvm = inv_gRretm.conj().T
        inv_gRadvp = inv_gRretp.conj().T

        # Build the retarded Green's function for the interface.
        HL_pr = HL + TL.conj().T @ la.solve(inv_gLretm, TL)
        HR_pr = HR + TR @ la.solve(inv_gRretp, TR.conj().T)
        # yapf: disable
        H_pr = np.block([
            [HL_pr, VLC, np.zeros((HL_pr.shape[0], HR_pr.shape[1]))],
            [VLC.conj().T, KC, VCR],
            [np.zeros((HR_pr.shape[0], HL_pr.shape[1])), VCR.conj().T, HR_pr]
        ])
        # yapf: enable
        Gret = la.pinv(en * np.eye(H_pr.shape[0], dtype=np.complex128) - H_pr,
                       rcond=1e-3)

        # Compute the total transmission.
        GammaL = 1.j * TL.conj().T @ (la.solve(inv_gLretm, TL) -
                                      la.solve(inv_gLadvm, TL))
        GammaR = 1.j * TR @ (la.solve(inv_gRretp,
                                      TR.conj().T) -
                             la.solve(inv_gRadvp,
                                      TR.conj().T))
        GLRret = Gret[:HL_pr.shape[0], -HR_pr.shape[1]:]
        GRLret = Gret[-HR_pr.shape[0]:, :HL_pr.shape[1]]
        trans[iomega] = np.trace(
            GammaR @ GRLret @ GammaL @ GRLret.conj().T).real

        # Build the Bloch matrices.
        FLretp = la.solve(inv_gLretp, TL.conj().T)
        FRretp = la.solve(inv_gRretp, TR.conj().T)
        FLadvp = la.solve(inv_gLadvp, TL.conj().T)
        FRadvp = la.solve(inv_gRadvp, TR.conj().T)
        inv_FLretm = la.solve(inv_gLretm, TL)
        inv_FRretm = la.solve(inv_gRretm, TR)
        inv_FLadvm = la.solve(inv_gLadvm, TL)
        inv_FRadvm = la.solve(inv_gRadvm, TR)

        def orthogonalize(values, vectors):
            modules = np.abs(values)
            phases = np.angle(values)
            order = np.argsort(-modules)
            values = np.copy(values[order])
            vectors = np.copy(vectors[:, order])
            lo = 0
            hi = 1
            groups = []
            while True:
                if hi >= vectors.shape[1] or not np.isclose(
                        values[hi], values[hi - 1], args.tolerance):
                    groups.append((lo, hi))
                    lo = hi
                if hi >= vectors.shape[1]:
                    break
                hi += 1
            for g in groups:
                lo, hi = g
                if hi > lo + 1:
                    values[lo:hi] = values[lo:hi].mean()
                    vectors[:, lo:hi] = la.orth(vectors[:, lo:hi])
            return values, vectors

        # Solve the corresponding eigenvalue equations for the leads.
        # Look for degenerate modes and orthonormalize them.
        ALretp, ULretp = orthogonalize(*la.eig(FLretp))
        ARretp, URretp = orthogonalize(*la.eig(FRretp))
        ALadvp, ULadvp = orthogonalize(*la.eig(FLadvp))
        ARadvp, URadvp = orthogonalize(*la.eig(FRadvp))
        ALretm, ULretm = orthogonalize(*la.eig(inv_FLretm))
        ARretm, URretm = orthogonalize(*la.eig(inv_FRretm))
        ALadvm, ULadvm = orthogonalize(*la.eig(inv_FLadvm))
        ARadvm, URadvm = orthogonalize(*la.eig(inv_FRadvm))

        # Find out which modes are propagating.
        mask_Lretp = np.isclose(np.abs(ALretp), 1., args.tolerance)
        mask_Rretp = np.isclose(np.abs(ARretp), 1., args.tolerance)
        mask_Ladvp = np.isclose(np.abs(ALadvp), 1., args.tolerance)
        mask_Radvp = np.isclose(np.abs(ARadvp), 1., args.tolerance)
        mask_Lretm = np.isclose(np.abs(ALretm), 1., args.tolerance)
        mask_Rretm = np.isclose(np.abs(ARretm), 1., args.tolerance)
        mask_Ladvm = np.isclose(np.abs(ALadvm), 1., args.tolerance)
        mask_Radvm = np.isclose(np.abs(ARadvm), 1., args.tolerance)

        # Compute the group velocity matrices.
        # yapf: disable
        VLretp = 1.j * aL * ULretp.conj().T @ TL @ (
            la.solve(inv_gLretp, TL.conj().T) -
            la.solve(inv_gLadvp, TL.conj().T)
        ) @ ULretp / 2. / omega
        VRretp = 1.j * aR * URretp.conj().T @ TR @ (
            la.solve(inv_gRretp, TR.conj().T) -
            la.solve(inv_gRadvp, TR.conj().T)
        ) @ URretp / 2. / omega
        VLadvp = 1.j * aL * ULadvp.conj().T @ TL @ (
            la.solve(inv_gLadvp, TL.conj().T) -
            la.solve(inv_gLretp, TL.conj().T)
        ) @ ULadvp / 2. / omega
        VRadvp = 1.j * aR * URadvp.conj().T @ TR @ (
            la.solve(inv_gRadvp, TR.conj().T) -
            la.solve(inv_gRretp, TR.conj().T)
        )  @ URadvp / 2. / omega
        VLretm = -1.j * aL * ULretm.conj().T @ TL.conj().T @ (
            la.solve(inv_gLretm, TL) -
            la.solve(inv_gLadvm, TL)
        ) @ ULretm / 2. / omega
        VRretm = -1.j * aR * URretm.conj().T @ TR.conj().T @ (
            la.solve(inv_gRretm, TR) -
            la.solve(inv_gRadvm, TR)
        ) @ URretm / 2. / omega
        VLadvm = -1.j * aL * ULadvm.conj().T @ TL.conj().T @ (
            la.solve(inv_gLadvm, TL) -
            la.solve(inv_gLretm, TL)
        ) @ ULadvm / 2. / omega
        VRadvm = -1.j * aR * URadvm.conj().T @ TR.conj().T @ (
            la.solve(inv_gRadvm, TR) -
            la.solve(inv_gRretm, TR)
        )  @ URadvm / 2. / omega
        # yapf: enable

        # Refine these matrices using the precomputed propagation masks.
        def refine(V, mask):
            diag = np.diag(V)
            nruter = np.zeros_like(diag)
            nruter[mask] = diag[mask].real
            return np.diag(nruter)

        VLretp = refine(VLretp, mask_Lretp)
        VRretp = refine(VRretp, mask_Rretp)
        VLadvp = refine(VLadvp, mask_Ladvp)
        VRadvp = refine(VRadvp, mask_Radvp)
        VLretm = refine(VLretm, mask_Lretm)
        VRretm = refine(VRretm, mask_Rretm)
        VLadvm = refine(VLadvm, mask_Ladvm)
        VRadvm = refine(VRadvm, mask_Radvm)

        # Set up auxiliary diagonal matrices with elements that are either
        # inverse of the previous diagonals or zero.
        def build_tilde(V):
            diag = np.diag(V)
            nruter = np.zeros_like(diag)
            indices = np.logical_not(np.isclose(np.abs(diag), 0.))
            nruter[indices] = 1. / diag[indices]
            return np.diag(nruter)

        VLretp_tilde = build_tilde(VLretp)
        VRretp_tilde = build_tilde(VRretp)
        VLadvp_tilde = build_tilde(VLadvp)
        VRadvp_tilde = build_tilde(VRadvp)
        VLretm_tilde = build_tilde(VLretm)
        VRretm_tilde = build_tilde(VRretm)
        VLadvm_tilde = build_tilde(VLadvm)
        VRadvm_tilde = build_tilde(VRadvm)

        # Build the matrices used to extract the transmission of the
        # perfect leads.
        ILretp = VLretp @ VLretp_tilde
        IRretp = VRretp @ VRretp_tilde
        ILadvp = VLadvp @ VLadvp_tilde
        IRadvp = VRadvp @ VRadvp_tilde
        ILretm = VLretm @ VLretm_tilde
        IRretm = VRretm @ VRretm_tilde
        ILadvm = VLadvm @ VLadvm_tilde
        IRadvm = VRadvm @ VRadvm_tilde

        # Compute the number of right-propagating and left-propagating
        # channels, and perform a small sanity check.
        NLp[iomega] = np.trace(ILretp).real
        NRp[iomega] = np.trace(IRretp).real
        NLm[iomega] = np.trace(ILretm).real
        NRm[iomega] = np.trace(IRretm).real

        oNLp = np.trace(ILadvp).real
        oNRp = np.trace(IRadvp).real
        oNLm = np.trace(ILadvm).real
        oNRm = np.trace(IRadvm).real
        if not (np.isclose(NLp[iomega], oNLp) and np.isclose(
                NRp[iomega], oNRp) and np.isclose(NLm[iomega], oNLm)
                and np.isclose(NRm[iomega], oNRm)):
            print("Warning: Inconsistent transmission for omega =", omega)

        NLp[iomega] = np.sum(mask_Lretp)
        NRp[iomega] = np.sum(mask_Rretp)
        NLm[iomega] = np.sum(mask_Ladvp)
        NRm[iomega] = np.sum(mask_Radvp)

        # Compute the matrices required for the transmission of the complete
        # system.
        QL = (en * np.eye(HL.shape[0], dtype=np.complex128) - HL -
              TL.conj().T @ la.solve(inv_gLretm, TL) -
              TL @ la.solve(inv_gLretp,
                            TL.conj().T))
        QR = (en * np.eye(HR.shape[0], dtype=np.complex128) - HR -
              TR.conj().T @ la.solve(inv_gRretm, TR) -
              TR @ la.solve(inv_gRretp,
                            TR.conj().T))
        VRretp12 = np.sqrt(VRretp)
        VLadvm12 = np.sqrt(VLadvm)
        VLretm12 = np.sqrt(VLretm)
        VRadvp12 = np.sqrt(VRadvp)

        tRL = 2.j * omega * (VRretp12 @ la.solve(URretp, GRLret) @ la.solve(
            ULadvm.conj().T, VLadvm12) / np.sqrt(aR * aL))
        tLR = 2.j * omega * (VLretm12 @ la.solve(ULretm, GLRret) @ la.solve(
            URadvp.conj().T, VRadvp12) / np.sqrt(aR * aL))

        #  Discard evanescent modes.
        tRL = tRL[mask_Rretp, :][:, mask_Ladvm]
        ALretp = ALretp[mask_Lretp]
        ARretp = ARretp[mask_Rretp]
        ALadvp = ALadvp[mask_Ladvp]
        ARadvp = ARadvp[mask_Radvp]
        ALretm = ALretm[mask_Lretm]
        ARretm = ARretm[mask_Rretm]
        ALadvm = ALadvm[mask_Ladvm]
        ARadvm = ARadvm[mask_Radvm]

        # Compute the total transmission again.
        trans_check[iomega] = np.diag(tRL.conj().T @ tRL).sum().real

    for iomega in range(1, inc_omega.size - 1):
        if not np.isclose(trans[iomega], trans_check[iomega], atol=1.):
            print("Problem at omega={} rad/ps".format(inc_omega[iomega]))

    ofilename = os.path.join(args.odir,
                             f"transmission_iface_{qa:.5f}_{qb:.5f}.txt")
    np.savetxt(ofilename, np.c_[inc_omega, trans, trans_check])
    print(f"Transmission saved to {ofilename:s}")
    ofilename = os.path.join(args.odir,
                             f"transmission_left_{qa:.5f}_{qb:.5f}.txt")
    np.savetxt(ofilename, np.c_[inc_omega, NLp])
    ofilename = os.path.join(args.odir,
                             f"transmission_right_{qa:.5f}_{qb:.5f}.txt")
    np.savetxt(ofilename, np.c_[inc_omega, NRp])

    if False:
        fig = plt.figure(figsize=(8., 10.))
        plt.plot(inc_omega, trans, label="Caroli")
        plt.plot(inc_omega, trans_check, label="detailed")
        plt.ylim(bottom=0.)
        plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
        plt.ylabel(r"$T(\omega)$")
        plt.legend(loc="best")
        plt.tight_layout()

        fig = plt.figure(figsize=(8., 10.))
        plt.subplot(221)
        plt.plot(inc_omega, NLm)
        plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
        plt.ylabel(r"$N_{L-}(\omega)$")
        plt.subplot(222)
        plt.plot(inc_omega, NRm)
        plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
        plt.ylabel(r"$N_{R-}(\omega)$")
        plt.subplot(223)
        plt.plot(inc_omega, NLp)
        plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
        plt.ylabel(r"$N_{L+}(\omega)$")
        plt.subplot(224)
        plt.plot(inc_omega, NRp)
        plt.xlabel("$\omega\;(\mathrm{rad/ps})$")
        plt.ylabel(r"$N_{R+}(\omega)$")
        plt.tight_layout()

        plt.show()
