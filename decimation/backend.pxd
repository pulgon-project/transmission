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

# Wrappers for the C++ code in backend.cpp
from libcpp cimport bool


cdef extern from "backend.hpp":
    cdef void q2omega_bridge(double complex[], double complex[], int,
                             double[], int, double[], double[])
    cdef void decimation_bridge(double complex[], double complex[], int,
                                double complex, double, int, double complex[])
