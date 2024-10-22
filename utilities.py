import numpy as np
from ipdb import set_trace
import ipdb
import logging
from pulgon_tools_wip.utils import fast_orth, get_character
import copy
import scipy.linalg as la
from phonopy.units import VaspToTHz
import scipy


def counting_y_from_xy(y,xy, direction=1, tolerance=1e-5):
    tmp1 = xy-y
    if direction>0:
        tmp2 = np.logical_and(tmp1[:, 1:] >= -tolerance, tmp1[:, :-1]<=tolerance)
    else:
        tmp2 = np.logical_and(tmp1[:, 1:] < tolerance, tmp1[:, :-1] >= -tolerance)
    counts = tmp2.sum()
    return counts


def get_adapted_matrix(DictParams, num_atom, matrices):
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
        DictParams
    )

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

        for kk in range(len(chara)):  # loop ops
            if IR_ndim==1:
                chara_conj = chara[kk].conj()
            else:
                # chara_conj = chara[kk].conj().trace()
                chara_conj = chara[kk].trace().conj()
            num_modes += chara_conj * matrices[kk].trace()
            projector += chara_conj * matrices[kk]
        projector = IR_ndim * projector / len(chara)
        # num_modes = (num_modes / len(chara)).real
        num_modes = projector.trace().real

        if np.isclose(num_modes, np.round(num_modes)):
            # num_modes = num_modes.astype(np.int32)
            num_modes = np.round(num_modes).astype(np.int32)
        else:
            set_trace()
            logging.ERROR("num_modes is not an integer")
        # basis, error = fast_orth(projector, num_modes)
        u, s, vh = scipy.linalg.svd(projector)

        basis = u[:,:num_modes]
        error = 1 - np.abs(s[num_modes - 1] - s[num_modes]) / np.abs(s[num_modes - 1])
        if error>0.05:
            print("error: ", error)
            set_trace()
        dimension.append(basis.shape[1])
        adapted.append(basis)

    adapted = np.concatenate(adapted, axis=1)
    if adapted.shape[0] != adapted.shape[1] or adapted.shape[0] != ndof:
        set_trace()
        logging.ERROR("the shape of adapted is incorrect")
    return adapted, dimension


def get_adapted_matrix_multiq(qpoints, DictParams, num_atom, matrices):
    """ The same with function:get_adapted_matrix but for multi-q points
    """

    adapteds, dimensions = [], []
    for qp in qpoints:
        DictParams.update({"qpoints": qp})
        characters, paras_values, paras_symbols = get_character(DictParams=DictParams)
        characters = np.array(characters)
        characters = characters[::2] + characters[1::2]   # combine all sigma(1,-1)

        idx_non_zero = np.nonzero(characters)
        characters[idx_non_zero[0], idx_non_zero[1]] = characters[idx_non_zero[0], idx_non_zero[1]] / np.abs(characters[idx_non_zero[0], idx_non_zero[1]])
        paras_values = paras_values[::2]
        res1 = np.round(characters @ characters.conj().T / characters.shape[1], 1)
        res2 = np.round(characters.conj().T @ characters / characters.shape[1], 1)

        ndof = 3 * num_atom
        remaining_dof = copy.deepcopy(ndof)
        adapted = []
        dimension = []
        for ii, chara in enumerate(characters):  # loop quantum number / irrep
            projector = np.zeros((ndof, ndof), dtype=np.complex128)
            prefactor = chara[0].real / len(chara)
            num_modes = 0
            for kk in range(len(chara)):  # loop ops
                projector += prefactor * chara[kk] * matrices[kk]
                num_modes += chara[kk].conj() * np.trace(matrices[kk])
                # projector += chara[kk] * matrices[kk]
            # set_trace()
            num_modes = (num_modes / len(chara)).real

            if num_modes.is_integer():
                num_modes = num_modes.astype(np.int32)
            # else:
            #     set_trace()
            #     logging.ERROR("num_modes is not an integer")
            basis, error = fast_orth(projector, remaining_dof, int(ndof / len(characters)))
            # basis, error = fast_orth(projector, remaining_dof, num_modes)
            adapted.append(basis)

            remaining_dof -= basis.shape[1]
            dimension.append(basis.shape[1])

        adapted = np.concatenate(adapted, axis=1)
        adapteds.append(adapted)
        dimensions.append(dimension)

    return adapteds, dimensions


def divide_irreps(vec, adapted, dimensions):
    """

    Args:
        vec:
        adapted:
        dimensions:

    Returns:

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
            # for tmp2 in tmp1:
            #     vectors.append((tmp2 * adapted)[:, start:end].sum(axis=1))
            # set_trace()
        start = copy.copy(end)
    means = np.array(means)
    if means.ndim > 1:
        means = means.T
    return np.array(means)


def divide_over_irreps(vecs, basis, dimensions, rcond=0.05):
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
    coeff = []
    for ii, b in enumerate(irrep_bases):
        combined_matrix = np.concatenate([vecs, -b], axis=1)
        # TODO: Handle the tolerance more sensibly and systematically.
        kernel = la.null_space(combined_matrix, rcond=rcond)
        n_solutions = kernel.shape[1]
        coefficients = kernel[:n_vecs, :]
        new_vecs = vecs @ coefficients
        if n_solutions > 0:
            new_vecs = la.orth(new_vecs)
        adapted_vecs.append(new_vecs)
        coeff.append(coefficients)
    found = sum(v.shape[1] for v in adapted_vecs)
    if found != n_vecs:
        res = divide_irreps(vecs.T,basis,dimensions)
        print(f"{n_vecs} were needed, but {found} were found")
        print("res:", res)
        set_trace()
        # res1 = np.linalg.svd(np.concatenate([vecs, -irrep_bases[0]], axis=1), full_matrices=True)[1]
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


def get_freqAndeigvec_from_qp(D, DictParams, num_atom, matrices):
    # DictParams = {"qpoints": qp, "nrot": nrot, "order": order_ops, "family": family, "a": aL}
    adapted, dimensions = get_adapted_matrix(DictParams, num_atom, matrices)
    D = adapted.conj().T @ D @ adapted
    start = 0
    tmp_band, tmp_vec = [], []
    for ir in range(len(dimensions)):
        end = start + dimensions[ir]
        block = D[start:end, start:end]

        # eig = la.eigvalsh(block)
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


def commuting(A, B):
    res = ((A @ B - B @ A)==0).all()
    return res

