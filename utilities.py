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


def unit_basis_complex_vectors(complex_vectors):
    if complex_vectors.ndim == 1:
        moduli = np.abs(complex_vectors)
        unit_complex_vectors = complex_vectors / moduli

        # set_trace()
        return unit_complex_vectors
    elif complex_vectors.ndim ==2:
        unit_complex_vectors = []
        for vec in complex_vectors:
            moduli = np.abs(vec)
            unit_complex_vectors.append(vec / moduli)
        unit_complex_vectors = np.array(unit_complex_vectors)
        return unit_complex_vectors
    else:
        print("Error: The dimension is not correct")


# def normalize_complex_vectors(complex_vectors):
#     magnitudes = np.sqrt(np.sum(np.abs(complex_vectors) ** 2, axis=1))
#     normalized_complex_vectors = complex_vectors / magnitudes[:, np.newaxis]
#     return normalized_complex_vectors


def get_adapted_matrix(qpoints, nrot, order, family, a, num_atom, matrices):
    adapteds, dimensions = [], []
    for qp in qpoints:
        characters, paras_values, paras_symbols = get_character([qp], nrot, order, family, a)
        characters = np.array(characters)
        characters = characters[::2] + characters[1::2]   # depend on the dimension of the character

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

            basis = fast_orth(projector, remaining_dof, int(ndof / len(characters)))
            adapted.append(basis)

            remaining_dof -= basis.shape[1]
            dimension.append(basis.shape[1])
        adapted = np.concatenate(adapted, axis=1)
        adapteds.append(adapted)
        dimensions.append(dimension)

        if adapted.shape[0] != adapted.shape[1]:
            print(ii, adapted.shape)
            print(dimension)
            set_trace()
            logging.ERROR("the shape of adapted not equal")
    return adapteds, dimensions


def devide_irreps(vec, adapted, dimensions):
    tmp1 = vec @ adapted
    start = 0
    means = []
    for im, dim in enumerate(dimensions):
        end = start + dim
        if vec.ndim == 1:
            means.append(np.abs(tmp1[start:end]).sum())

        else:
            means.append(np.abs(tmp1[:, start:end]).sum(axis=1))
        start = copy.copy(end)
    means = np.array(means)
    if means.ndim > 1:
        means = means.T
    return np.array(means)

def divide_irreps2(vec, adapted, dimensions):
    tmp1 = vec @ adapted.conj()
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


def refine_qpoints(values, tol=1e-2):
    modules = np.abs(values)
    order = np.argsort(modules)
    values = np.copy(values[order])
    lo = 0
    hi = 1
    groups = []

    while True:
        if hi >= values.shape[0] or not np.isclose(
                values[hi], values[hi - 1], tol
        ):
            groups.append((lo, hi))
            lo = hi
        if hi > len(values):
            break
        hi += 1
    for g in groups:
        lo, hi = g
        if hi > lo + 1:
            values[lo:hi] = values[lo:hi].mean()
    return values, order


def combination_paras(vectors, adapted, means, dimensions, tol=1e-3):
    if len(np.unique(dimensions)) != 1:
        logging.Error("the length of dimension not equal")
    dim = dimensions[0]

    idx1 = means
    itp_basis = np.unique(np.where(idx1)[1])
    # if idx1[0].sum()==1:
    if len(itp_basis)==1:
        irreps = np.where(idx1[0])[0].repeat(vectors.shape[0])
        itp1 = itp_basis * dim
        itp2 = itp1 + dim
        proj = vectors @ adapted
        paras = la.pinv(proj[:, itp1.item(): itp2.item()])
        paras = paras / np.linalg.norm(paras, axis=1)[:, np.newaxis]    # normalization

        res = paras @ vectors
        U_irreps = res[0]
        means = devide_irreps(U_irreps, adapted, dimensions)
#        paras = paras[0]   # row vector
        paras = np.eye(vectors.shape[0])

    else:
        # itp1 = np.where(idx1[0])[0] * dim
        itp1 = itp_basis * dim
        itp2 = itp1 + dim
        proj = vectors @ adapted

        for it, (tmp1, tmp2) in enumerate(zip(itp1, itp2)):
            if it == 0:
                tmp_matrix = proj[:, tmp1:tmp2]
            else:
                tmp_matrix = np.hstack((tmp_matrix, proj[:, tmp1:tmp2]))
        paras = la.pinv(tmp_matrix)
        paras = paras / np.linalg.norm(paras, axis=1)[:,np.newaxis]   # normalization

        paras = paras[::dim]   # row vector
        res = paras @ vectors

        U_irreps = res
        proj1 = U_irreps @ adapted
        start = 0
        means = []
        for im, _ in enumerate(dimensions):
            end = start + dim
            means.append(np.abs(proj1[:, start:end]).sum(axis=1))
            start = copy.copy(end)
        means = np.array(means)
        idx2 = (np.abs(means) > tol).T
        irreps = np.where(idx2)[1]
        # if len(irreps)!=vectors.shape[0]:
        #     set_trace()
        #     logging.ERROR("the shape of irreps incorect")
    return irreps, paras, means

