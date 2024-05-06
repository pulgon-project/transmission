import numpy as np
from ipdb import set_trace
import logging
from pulgon_tools_wip.utils import fast_orth, get_character
import copy
import scipy.linalg as la

def counting_y_from_xy(y,xy, direction=1, tolerance=1e-5):
    tmp1 = xy-y
    if direction>0:
        tmp2 = np.logical_and(tmp1[:, 1:] >= -tolerance, tmp1[:, :-1]<=tolerance)
    else:
        tmp2 = np.logical_and(tmp1[:, 1:] < tolerance, tmp1[:, :-1] >= -tolerance)
    counts = tmp2.sum()
    return counts


def get_adapted_matrix(qp, nrot, order, family, a, num_atom, matrices):
    """

    Args:
        qp: q/k point
        nrot: The rotational quantum number of the structure "Cn"
        order: The multiplication order from generators to symmetry operations
        family: The line group family index
        a: The period length in z direction
        num_atom: The number of atoms
        matrices:

    Returns:
        adapted: The symmetry projection basis matrix for irreducible representation

    """
    characters, paras_values, paras_symbols = get_character(
        [qp], nrot, order, family, a
    )
    characters = np.array(characters)
    characters = (
        characters[::2] + characters[1::2]
    )  # depend on the dimension of the character

    ndof = 3 * num_atom
    remaining_dof = copy.deepcopy(ndof)
    adapted = []
    dimension = []
    for ii, chara in enumerate(characters):  # loop quantum number
        projector = np.zeros((ndof, ndof), dtype=np.complex128)
        # prefactor = chara[0].real / len(chara)
        for kk in range(len(chara)):  # loop ops
            # projector += prefactor * chara[kk] * matrices[kk]
            projector += chara[kk] * matrices[kk]
        basis, error = fast_orth(projector, remaining_dof, int(ndof / len(characters)))
        # print("qp:", qp)
        # print("error", error)
        
        adapted.append(basis)
        remaining_dof -= basis.shape[1]
        dimension.append(basis.shape[1])
    adapted = np.concatenate(adapted, axis=1)

    return adapted, dimension


def get_adapted_matrix_multiq(qpoints, nrot, order, family, a, num_atom, matrices):
    """ The same with function:get_adapted_matrix but for multi-q points
    """
    adapteds, dimensions, chas = [], [], []
    for qp in qpoints:
        characters, paras_values, paras_symbols = get_character([qp], nrot, order, family, a)
        characters = np.array(characters)

        characters = characters[::2] + characters[1::2]   # combine all sigma(1,-1)
        # characters = characters[::2]   # combine all sigma(1,-1)
        paras_values = paras_values[::2]
        chas.append(characters)

        ndof = 3 * num_atom
        remaining_dof = copy.deepcopy(ndof)
        adapted = []
        dimension = []

        for ii, chara in enumerate(characters):  # loop quantum number
            projector = np.zeros((ndof, ndof), dtype=np.complex128)
            prefactor = chara[0].real / len(chara)
            for kk in range(len(chara)):  # loop ops
                projector += prefactor * chara[kk] * matrices[kk]
                # projector += chara[kk] * matrices[kk]
            basis, error = fast_orth(projector, remaining_dof, int(ndof / len(characters)))
            adapted.append(basis)

            remaining_dof -= basis.shape[1]
            dimension.append(basis.shape[1])

        adapted = np.concatenate(adapted, axis=1)
        adapteds.append(adapted)
        dimensions.append(dimension)

        print("qpoint:", qp)
        # print("error:", error)
        _, s, _ = la.svd(projector)
        # val = np.abs(val)
        # val = val[np.argsort(-val)]
        print("singular values: ", s[:(int(ndof / len(characters)) + 1)])
        # set_trace()

    return adapteds, dimensions


def divide_irreps(vec, adapted, dimensions):
    """

    Args:
        vec:
        adapted:
        dimensions:

    Returns:

    """
    # tmp1 = vec @ adapted.conj()
    tmp1 = vec @ adapted
    start = 0
    means, vectors = [], []
    for im, dim in enumerate(dimensions):
        end = start + dim
        if vec.ndim == 1:
            means.append((np.abs(tmp1[start:end]) ** 2).sum())
            # vectors.append((tmp1 * adapted)[:, start:end].sum(axis=1))
        else:
            means.append((np.abs(tmp1[:, start:end]) ** 2).sum(axis=1))
            # for tmp2 in tmp1:
            #     vectors.append((tmp2 * adapted)[:, start:end].sum(axis=1))
            # set_trace()
        start = copy.copy(end)
    means = np.array(means)
    if means.ndim > 1:
        means = means.T
    return np.array(means)


def divide_over_irreps(vecs, basis, dimensions):
    """

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

    for b in irrep_bases:
        combined_matrix = np.concatenate([vecs, -b], axis=1)
        # TODO: Handle the tolerance more sensibly and systematically.
        kernel = la.null_space(combined_matrix, rcond=1e-1)
        # set_trace()
        n_solutions = kernel.shape[1]
        coefficients = kernel[:n_vecs, :]
        new_vecs = vecs @ coefficients
        if n_solutions > 0:
            new_vecs = la.orth(new_vecs)
        adapted_vecs.append(new_vecs)
    found = sum(v.shape[1] for v in adapted_vecs)
    if found != n_vecs:
        # res = divide_irreps(vecs.T,basis,dimensions)
        raise ValueError(f"{n_vecs} were needed, but {found} were found")
    return adapted_vecs


def check_same_space(array1, array2):
    """ Check whether these two vectors from the same space or not
    """
    rank_array1 = np.linalg.matrix_rank(array1)
    rank_array2 = np.linalg.matrix_rank(array2)

    if rank_array1 == rank_array2:   # if the rank are the same, then check the rank of combine matrix
        space1 = np.linalg.matrix_rank(np.hstack([array1, array2]))
        space2 = np.linalg.matrix_rank(np.hstack([array2, array1]))

        if space1 == space2 == rank_array1:
            return True
    return False

