import numpy as np
from ipdb import set_trace
import ipdb
import logging
from pulgon_tools_wip.utils import fast_orth, get_character
import copy
import scipy.linalg as la
from phonopy.units import VaspToTHz
import scipy
from sympy.physics.quantum import TensorProduct


def get_branchNum_from_freq(y,xy, direction=1, tolerance=1e-5):
    """
    Counts the number of phonon at frequency y

    Parameters
    ----------
    y : array_like
        frequency
    xy : array_like
        The points to be counted
    direction : int, optional
        If direction is 1, counts the points to the right of y. If direction is -1, counts the points to the left of y. Default is 1
    tolerance : float, optional
        The tolerance for the comparison. Default is 1e-5

    Returns
    -------
    counts : int
        The number of phonon in xy that are within the tolerance of y in the x direction
    """
    tmp1 = xy-y
    if direction>0:
        tmp2 = np.logical_and(tmp1[:, 1:] >= -tolerance, tmp1[:, :-1]<=tolerance)
    else:
        tmp2 = np.logical_and(tmp1[:, 1:] < tolerance, tmp1[:, :-1] >= -tolerance)
    counts = tmp2.sum()
    return counts


def get_adapted_matrix(DictParams, num_atom, matrices):
    """
    Calculate the symmetry projection basis matrix for irreducible representation

    Parameters
    ----------
    DictParams : dict
        Dictionary of parameters:
           {"qp": q/k point | float,
            "nrot": The rotational quantum number of the structure "Cn" | int,
            "order": The multiplication order from generators to symmetry operations | list,
            "family": The line group family index | int,
            "a": The period length in z direction | float}
    num_atom : int
        The number of atoms
    matrices : list of numpy arrays
        The matrices used to calculate the symmetry projection basis matrix

    Returns
    -------
    adapted : numpy array
        The symmetry projection basis matrix for irreducible representation
    dimension : list of int
        The dimensions of the irreducible representations
    """
    characters, paras_values, paras_symbols = get_character(DictParams)
    ndof = 3 * num_atom
    adapted = []
    dimension = []
    for ii, chara in enumerate(characters):  # loop IR
        if chara.ndim ==1:
            IR_ndim = 1
        else:
            IR_ndim = chara.shape[-1]
        num_modes = 0
        projector = np.zeros((ndof, ndof), dtype=np.complex128)
        for kk in range(len(chara)):
            if IR_ndim==1:
                chara_conj = chara[kk].conj()
            else:
                chara_conj = chara[kk].trace().conj()
            num_modes += chara_conj * matrices[kk].trace()
            projector += chara_conj * matrices[kk]
        projector = IR_ndim * projector / len(chara)
        # num_modes = (num_modes / len(chara)).real     # Another way to get the dimention of the IR
        num_modes = projector.trace().real

        if np.isclose(num_modes, np.round(num_modes)):
            num_modes = np.round(num_modes).astype(np.int32)
        else:
            print("num_modes=", num_modes)
            logging.ERROR("num_modes is not an integer")
        u, s, vh = scipy.linalg.svd(projector)

        basis = u[:,:num_modes]
        error = 1 - np.abs(s[num_modes - 1] - s[num_modes]) / np.abs(s[num_modes - 1])
        if error>0.05:
            logging.ERROR("the error of svd is too large, error=", error)
            # print("error: ", error)
        dimension.append(basis.shape[1])
        adapted.append(basis)
    adapted = np.concatenate(adapted, axis=1)
    if adapted.shape[0] != adapted.shape[1] or adapted.shape[0] != ndof:
        logging.ERROR("the shape of adapted is incorrect, adapted.shape=", adapted.shape)
        # print("adapted.shape=", adapted.shape)
    return adapted, dimension


def divide_irreps(vec, adapted, dimensions):
    """
    Project vectors into IR space according to the adapted basis.

    Parameters
    ----------
    vec : array of shape (n,m), where n is the number of vectors and m is the dimension
        The vector to be divided
    adapted : array of shape (m,m). The basis vectors are arranged in columns.
        The adapted basis
    dimensions : list of int. Sum(dimensions) = m.
        The dimension of each irrep

    Returns
    -------
    means : array of shape (n,k), where k is the number of irreps
        The projected length of each irrep
    """
    tmp1 = vec @ adapted.conj()
    # tmp1 = vec @ adapted
    start = 0
    means, vectors = [], []
    for im, dim in enumerate(dimensions):
        end = start + dim
        if vec.ndim == 1:
            means.append((np.abs(tmp1[start:end]) ** 2).sum())
            # vectors.append((tmp1 * adapted)[:, start:end].sum(axis=1))
        else:
            means.append((np.abs(tmp1[:, start:end]) ** 2).sum(axis=1))
        start = copy.copy(end)
    means = np.array(means)
    if means.ndim > 1:
        means = means.T
    return np.array(means)


def divide_over_irreps(vecs, basis, dimensions, rcond=0.05):
    """
    Divide a set of vectors over irreps.

    Args:
        vecs: original eigenvectors spanning different irreps
        basis: the symmetry-adapted basis
        dimensions: the dimension of each irrep

    Returns:
        adapted_vecs: new eigenvectors, each of them spanning only one irrep

    Note:
        The solution is not unique, this function returns one possible solution.
    """
    n_vecs = vecs.shape[1]
    splits = np.cumsum(dimensions)[:-1]
    irrep_bases = np.split(basis, splits, axis=1)
    adapted_vecs = []
    coeff = []
    for ii, b in enumerate(irrep_bases):
        combined_matrix = np.concatenate([vecs, -b], axis=1)
        kernel = la.null_space(combined_matrix, rcond=rcond)
        n_solutions = kernel.shape[1]
        # coefficients = kernel[:n_vecs, :]
        # new_vecs = vecs @ coefficients
        coefficients = kernel[n_vecs:, :]
        new_vecs = -b @ coefficients

        if n_solutions > 0:
            new_vecs = la.orth(new_vecs)
        adapted_vecs.append(new_vecs)
        coeff.append(coefficients)
    found = sum(v.shape[1] for v in adapted_vecs)
    if found != n_vecs:
        raise ValueError(f"{n_vecs} were needed, but {found} were found")
    return adapted_vecs


def divide_over_irreps_using_projectors(vecs, basis, dimensions):
    """
    Divide a set of vectors over irreps using projectors.

    Args:
        vecs: original eigenvectors spanning different irreps
        basis: the symmetry-adapted basis
        dimensions: the dimension of each irrep

    Returns:
        adapted_vecs: new eigenvectors, each of them spanning only one irrep
    """
    n_vecs = vecs.shape[1]
    splits = np.cumsum(dimensions)[:-1]
    irrep_bases = np.split(basis, splits, axis=1)
    adapted_vecs = []
    projections = []
    for b in irrep_bases:
        projector = sum(
            np.outer(b[:, i], b[:, i].T.conj()) for i in range(b.shape[1])
        )
        restrictions = projector @ vecs
        new_vecs = la.orth(restrictions)
        adapted_vecs.append(new_vecs)
        new_projections = new_vecs.conj().T @ vecs
        new_projections = (new_projections.conj() * new_projections).real.sum(
            axis=-1)
        projections.append(new_projections)

    all_projections = []
    for p in projections:
        all_projections.extend(p.tolist())
    best_indices = (-np.asarray(all_projections)).argsort()
    selected = best_indices[:n_vecs].tolist()

    i_total = 0
    selected_vecs = []
    for i_r, projs in enumerate(projections):
        new_vecs = []
        for i_p, p in enumerate(projs):
            if i_total in selected:
                new_vecs.append(adapted_vecs[i_r][:, i_p].reshape(-1,1))
                # selected_vecs.append(adapted_vecs[i_r][:, i_p])
            i_total += 1

        if len(new_vecs) > 1:
            new_vecs = np.concatenate(new_vecs, axis=1)
        elif len(new_vecs) == 1:
            new_vecs = new_vecs[0]
        else:
            new_vecs = np.array(new_vecs)
        selected_vecs.append(new_vecs)
    # selected_vecs = [np.asarray(s) for s in selected_vecs]
    return selected_vecs


def check_same_space(array1, array2):
    """
    Check whether these two vectors from the same space or not.

    Parameters
    ----------
    array1 : array_like
        The first array.
    array2 : array_like
        The second array.

    Returns
    -------
    bool
        True if the arrays are from the same space, False otherwise.
    """
    rank_array1 = np.linalg.matrix_rank(array1)
    rank_array2 = np.linalg.matrix_rank(array2)

    if rank_array1 == rank_array2:   # if the rank are the same, then check the rank of combine matrix
        space1 = np.linalg.matrix_rank(np.hstack([array1, array2]))
        space2 = np.linalg.matrix_rank(np.hstack([array2, array1]))

        if space1 == space2 == rank_array1:
            return True
    return False


def get_freqAndeigvec_from_qp(D, DictParams, num_atom, matrices):
    """
    Calculate the frequencies and eigenvectors at a given qpoint
    by using the symmetry-adapted basis.

    Parameters
    ----------
    D : array_like
        The dynamical matrix at the given qpoint.
    DictParams : dict
        A dictionary of parameters, including "qpoints", "nrot", "order", "family", and "a".
    num_atom : int
        The number of atoms.
    matrices : list of numpy arrays
        The representation matrices used to calculate the symmetry projection basis matrix

    Returns
    -------
    freq : numpy array
        The frequencies at the given qpoint.
    eigvec : numpy array
        The eigenvectors at the given qpoint.
    """
    adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)
    D = adapted.conj().T @ D @ adapted
    start = 0
    tmp_band, tmp_vec = [], []
    for ir in range(len(dimensions)):
        end = start + dimensions[ir]
        block = D[start:end, start:end]
        eig, vec = la.eigh(block)
        e = (
                np.sqrt(np.abs(eig))
                * np.sign(eig)
                * VaspToTHz
            # * phonopy.units.VaspToEv
            # * 1e3
        ).tolist()
        tmp_band.append(e)
        tmp_vec.append(vec)
        start = end
    tmp_band = np.array(tmp_band) * 2 * np.pi
    tmp_vec = np.array(tmp_vec)
    return np.array(tmp_band)  , np.array(tmp_vec)


def get_modified_adapted_matrix(DictParams, num_atom, matrices):
    """
    Calculate the modified symmetry adapted basis matrix for irreducible representation using the modified algorithm.

    Parameters
    ----------
    DictParams : dict
        A dictionary of parameters, including "qpoints", "nrot", "order", "family", and "a".
    num_atom : int
        The number of atoms.
    matrices : list of numpy arrays
        The matrices used to calculate the symmetry projection basis matrix.

    Returns
    -------
    adapted : numpy array
        The symmetry projection basis matrix for irreducible representation.
    dimension : list of int
        The dimensions of the irreducible representations
    """
    characters, paras_values, paras_symbols = get_character(
        DictParams
    )
    adapted = []
    dimension = []
    for ii, chara in enumerate(characters):  # loop IR
        if chara.ndim ==1:
            IR_ndim = 1
        else:
            IR_ndim = chara.shape[-1]
        num_modes = 0

        ndof = IR_ndim * matrices[0].shape[0]
        projector = np.zeros((ndof, ndof), dtype=np.complex128)
        for kk in range(len(chara)):  # loop ops
            projector += TensorProduct(np.array(chara[kk]), matrices[kk])

        projector = projector / len(chara)
        # num_modes = (num_modes / len(chara)).real
        num_modes = projector.trace().real

        if np.isclose(num_modes, np.round(num_modes)):
            num_modes = np.round(num_modes).astype(np.int32)
        else:
            logging.ERROR("num_modes is not an integer")

        u, s, vh = scipy.linalg.svd(projector)
        basis = u[:,:num_modes]
        error = 1 - np.abs(s[num_modes - 1] - s[num_modes]) / np.abs(s[num_modes - 1])
        if error>0.05:
            logging.ERROR("the error of svd is too large, error=", error)
        if IR_ndim != 1:
            basis = np.array(np.array_split(basis, 2, axis=0))
            basis = scipy.linalg.orth(np.concatenate(basis, axis=1))
        dimension.append(basis.shape[1])
        adapted.append(basis)
    adapted = np.concatenate(adapted, axis=1)
    if adapted.shape[0] != adapted.shape[1]:
        logging.ERROR("the shape of adapted is incorrect, shape=", adapted.shape)
    return adapted, dimension




