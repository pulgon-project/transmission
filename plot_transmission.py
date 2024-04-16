#!/usr/bin/env python

import os
import os.path
import sys
import glob

import numpy as np
import scipy as sp
import scipy.integrate
import scipy.linalg as la
import matplotlib
import matplotlib.pyplot as plt

import gzip
import lzma
import tqdm
import ase
import ase.io

matplotlib.rcParams["font.size"] = 16.
# Physical constants.
kB = 1.3806503e-23  # J/K
hbar = 1.05457172647e-22  # J s/rad


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


def are_close(a, b, *args, **kwargs):
    """Return true if two reduced coordinate vectors are compatible."""
    delta = a - b
    delta -= np.round(delta)
    return np.isclose(la.norm(delta), 0., *args, **kwargs)


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


def calc_kernel(omega, T):
    """Compute the dimensionless part of the B-E derivative."""
    LO = 1e-6
    HI = 20.
    x = hbar * omega / (2. * kB * T)
    nruter = np.empty_like(omega)
    between = np.logical_and(x > LO, x < HI)
    nruter[between] = (x[between] / np.sinh(x[between]))**2
    nruter[x >= HI] = 0.
    nruter[x <= LO] = 1.
    return nruter


def calc_GT(omega, transmission, T, A):
    """Compute the conductance derived from a transmission."""
    kernel = calc_kernel(omega, T)
    return kB * sp.integrate.trapz(
        y=kernel * transmission, x=omega * 1e12) / 2. / np.pi / A


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: {} coordinate_file directory".format(sys.argv[0]))
    poscar = sys.argv[1]
    directory = sys.argv[2]

    try:
        poscar = ase.io.read(poscar)
    except ValueError:
        poscar = read_lammps(poscar)
    nx = find_cell_along_axis(poscar, 0, 1e-4)
    ny = find_cell_along_axis(poscar, 1, 1e-4)
    print("Parallel supercell size: {} x {}".format(nx, ny))
    lattvec = poscar.get_cell() * 1e-10  # m
    A = la.det(lattvec[:2, :2]) / float(nx * ny)  # m ** 2

    fns_iface = glob.glob(os.path.join(directory, "transmission_iface_*.txt"))
    fns_left = glob.glob(os.path.join(directory, "transmission_left_*.txt"))
    fns_right = glob.glob(os.path.join(directory, "transmission_right_*.txt"))
    # yapf: disable
    fns_intersection = (
        set("_".join(os.path.basename(i).split("_")[-2:]) for i in fns_iface) &
        set("_".join(os.path.basename(i).split("_")[-2:]) for i in fns_left) &
        set("_".join(os.path.basename(i).split("_")[-2:]) for i in fns_right))
    # yapf: enable
    pattern = np.loadtxt(fns_iface[0])
    transmission_iface = np.zeros_like(pattern)
    transmission_left = np.zeros_like(pattern)
    transmission_right = np.zeros_like(pattern)
    transmission_iface[:, 0] = pattern[:, 0]
    transmission_left[:, 0] = pattern[:, 0]
    transmission_right[:, 0] = pattern[:, 0]
    processed = 0
    for fn in fns_intersection:
        data = np.loadtxt(os.path.join(directory, "transmission_iface_" + fn))
        if not np.allclose(data[:, 1], data[:, 2], atol=1.):
            print("Problem with:", fn)
        processed += 1
        transmission_iface[:, 1] += data[:, 1]
        data = np.loadtxt(os.path.join(directory, "transmission_left_" + fn))
        transmission_left[:, 1] += data[:, 1]
        data = np.loadtxt(os.path.join(directory, "transmission_right_" + fn))
        transmission_right[:, 1] += data[:, 1]
    transmission_iface[:, 1] /= processed
    transmission_left[:, 1] /= processed
    transmission_right[:, 1] /= processed

    np.savetxt(
        os.path.join(directory, "average_transmission.txt"),
        np.c_[transmission_left[:, 0], transmission_left[:, 1],
              transmission_iface[:, 1], transmission_right[:, 1]],
        header="omega[rad/ps]\tT_left\tT_interface\tT_right")

    transmission_iface[:, 1] = transmission_iface[:, 1].clip(min=0.)
    omega = transmission_iface[:, 0]
    Tvec = np.linspace(10., 500., num=100, endpoint=True)
    Rint = np.empty_like(Tvec)
    RL_vec = np.empty_like(Tvec)
    RR_vec = np.empty_like(Tvec)
    RT_vec = np.empty_like(Tvec)
    for i, T in enumerate(Tvec):
        GT = calc_GT(omega, transmission_iface[:, 1], T, A)
        RT = 1. / GT
        RT_vec[i] = RT
        GL = calc_GT(omega, transmission_left[:, 1], T, A)
        RL = 1. / GL
        RL_vec[i] = RL
        GR = calc_GT(omega, transmission_right[:, 1], T, A)
        RR = 1. / GR
        RR_vec[i] = RR
        Rint[i] = RT - .5 * (RL + RR)

    np.savetxt(
        os.path.join(directory, "interface_resistance.txt"),
        np.c_[Tvec, Rint, RT_vec, RL_vec, RR_vec],
        header="T[K]\tRint[m^2K/W]\tRT[m^2K/W]\tRL[m^2K/W]\tRR[m^2K/W]")

    fig = plt.figure()
    plt.subplot(131)
    plt.plot(transmission_left[:, 0], transmission_left[:, 1])
    plt.ylim(-0.1, 2.1)
    plt.xlabel(r"$\omega\;(\mathrm{rad / ps})$")
    plt.ylabel(r"$\left\langle T\left(\omega\right)\right\rangle$")
    plt.gca().set_title("Left lead")
    plt.subplot(132)
    plt.plot(transmission_iface[:, 0], transmission_iface[:, 1])
    plt.ylim(-0.1, 2.1)
    plt.gca().set_yticklabels([])
    plt.xlabel(r"$\omega\;(\mathrm{rad / ps})$")
    plt.gca().set_title("Interface")
    plt.subplot(133)
    plt.plot(transmission_right[:, 0], transmission_right[:, 1])
    plt.ylim(-0.1, 2.1)
    plt.gca().set_yticklabels([])
    plt.xlabel(r"$\omega\;(\mathrm{rad / ps})$")
    plt.gca().set_title("Right lead")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    plt.savefig(
        os.path.join(directory, "transmission.pdf"), bbox_inches="tight")

    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()

    fig.canvas.mpl_connect("resize_event", on_resize)

    fig2 = plt.figure()
    plt.plot(transmission_left[:, 0], transmission_left[:, 1], label="left")
    plt.plot(
        transmission_iface[:, 0], transmission_iface[:, 1], label="interface")
    plt.plot(transmission_right[:, 0], transmission_right[:, 1], label="right")
    plt.ylim(-0.1, 2.1)
    plt.xlabel(r"$\omega\;(\mathrm{rad / ps})$")
    plt.ylabel(r"$\left\langle T\left(\omega\right)\right\rangle$")
    plt.legend(loc="best")
    plt.tight_layout()

    def on_resize(event):
        fig2.tight_layout()
        fig2.canvas.draw()

    fig2.canvas.mpl_connect("resize_event", on_resize)

    plt.figure()
    plt.plot(Tvec, Rint)
    ax = plt.gca()
    ax.set_yscale("log")
    # plt.ylim(bottom=1.e-10, top=1e-9)
    plt.xlabel(r"$T\;\left(\mathrm{K}\right)$")
    plt.ylabel(r"$R_T\;\left(\mathrm{m^2K/W}\right)$")
    plt.savefig(os.path.join(directory, "resistance.pdf"), bbox_inches="tight")

    plt.tight_layout()
    plt.show()
