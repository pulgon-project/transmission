// Copyright 2018 Jesús Carrete Montaña <jesus.carrete.montana@tuwien.ac.at>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

#pragma once

#include <complex>


void q2omega_bridge(const std::complex<double>* hh0,
                    const std::complex<double>* tt0,
                    int ndof,
                    const double* q0,
                    int nq,
                    double* omega0,
                    double* vg0);


void decimation_bridge(const std::complex<double>* k00in,
                       const std::complex<double>* k01in,
                       int ndof,
                       std::complex<double> en,
                       double tolerance,
                       int maxiter,
                       std::complex<double>* inv_g00out);
