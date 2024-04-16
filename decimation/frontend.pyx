# Copyright 2018 Jesús Carrete Montaña <jesus.carrete.montana@tuwien.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# cython: boundscheck=False
# cython: cdivision=True

import numpy as np
import scipy as sp
import scipy.linalg as la

from decimation.backend cimport q2omega_bridge
from decimation.backend cimport decimation_bridge
cimport numpy as np

np.import_array()

# Note that, when compiling as C++, "double complex" is treated by Cython
# as std::complex<double>.


def q2omega(hh, tt, qs):
    """Compute the frequencies and wave functions at a set of wave numbers."""
    cdef int ndof = hh.shape[0]
    if hh.ndim != 2 or hh.shape[1] != ndof:
        raise ValueError("hh must be a square matrix")
    if tt.ndim != 2 or tt.shape != (ndof, ndof):
        raise ValueError("tt must be a square matrix with the same"
                         "shape as hh")
    cdef double complex[:, :] c_hh = np.asfortranarray(hh, dtype=np.complex128)
    cdef double complex[:, :] c_tt = np.asfortranarray(tt, dtype=np.complex128)
    cdef int nq = qs.size
    cdef double[:] c_qs = np.ascontiguousarray(qs)
    omega = np.empty((nq, ndof), order="C")
    cdef double[:, :] c_omega = omega
    vg = np.empty((nq, ndof), order="C")
    cdef double[:, :] c_vg = vg
    q2omega_bridge(&(c_hh[0, 0]), &(c_tt[0, 0]), ndof,
                  &(c_qs[0]), nq, &(c_omega[0, 0]), &(c_vg[0, 0]))
    return (omega, vg)


def inv_g00(hh, tt, omega, eps, tolerance, maxiter):
    """Compute the block of the GF of a lead using decimation and return its
    inverse.
    """
    cdef int ndof = hh.shape[0]
    if hh.ndim != 2 or hh.shape[1] != ndof:
        raise ValueError("hh must be a square matrix")
    if tt.ndim != 2 or tt.shape != (ndof, ndof):
        raise ValueError("tt must be a square matrix with the same"
                         "shape as hh")
    cdef double complex en = omega * (omega + 1.j * eps)
    cdef double complex[:, :] c_hh = np.asfortranarray(hh, dtype=np.complex128)
    cdef double complex[:, :] c_tt = np.asfortranarray(tt.conj().T,
                                                       dtype=np.complex128)
    ws = np.empty_like(hh, order="F")
    cdef double complex[:, :] c_ws = ws
    decimation_bridge(&(c_hh[0, 0]), &(c_tt[0, 0]), ndof, en, tolerance,
                      maxiter, &(c_ws[0, 0]))
    # Note the change of sign required for consistency with the rest of the
    # formalism.
    return -ws
