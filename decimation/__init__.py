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

from decimation.frontend import q2omega
from decimation.frontend import inv_g00

import numpy as np
import scipy as sp
import scipy.linalg as la


def common_values(v1, v2):
    nruter = 0
    for i in v1:
        if np.isclose(i, 0.):
            continue
        for j in v2:
            if np.isclose(i, j):
                nruter += 1
                break
    return nruter


def _unique_angles(angles, *args, **kwargs):
    """Find an unique set of angles among the input, where uniqueness is
    determined using np.isclose, and 2 * pi periodicity is taken into accound.
    """
    starting = np.sort(angles)
    last = None
    nruter = []
    for s in starting:
        if last is None:
            nruter.append(s)
        else:
            delta = (s - last) / 2. / np.pi
            delta -= round(delta)
            if not np.isclose(delta, 0., *args, **kwargs):
                nruter.append(s)
        last = s
    return nruter
