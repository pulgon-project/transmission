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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "backend.hpp"

constexpr std::complex<double> iota(0., 1.);


class Decimation_MaxIterException : public std::exception {
    const char* what() const noexcept
    {
        return "the maximum number of iterations was reached";
    }
};


void q2omega_bridge(const std::complex<double>* hh0,
                    const std::complex<double>* tt0,
                    int ndof,
                    const double* q0,
                    int nq,
                    double* omega0,
                    double* vg0)
{
    Eigen::Map<const Eigen::MatrixXcd> hh(hh0, ndof, ndof);
    Eigen::Map<const Eigen::MatrixXcd> tt(tt0, ndof, ndof);
    Eigen::Map<const Eigen::ArrayXd> qs(q0, nq);
    Eigen::Map<Eigen::MatrixXd> omega(omega0, ndof, nq);
    Eigen::Map<Eigen::MatrixXd> vg(vg0, ndof, nq);

    Eigen::ArrayXcd expfactors = (-iota * qs).exp();
    Eigen::MatrixXcd D(ndof, ndof);
    Eigen::MatrixXcd dDdk(ndof, ndof);
    for (int iq = 0; iq < nq; ++iq) {
        D = hh + tt * expfactors(iq) + tt.adjoint() * std::conj(expfactors(iq));
        dDdk = iota *
               (tt * expfactors(iq) - tt.adjoint() * std::conj(expfactors(iq)));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(D);
        Eigen::ArrayXd omega2 = D.eigenvalues().real().array();
        omega.col(iq) = omega2.sign() * omega2.abs().sqrt();
        Eigen::MatrixXcd phi = solver.eigenvectors();
        for (int alpha = 0; alpha < ndof; ++alpha) {
            if (std::abs(omega(alpha, iq)) < 1e-5) {
                omega(alpha, iq) = 0.;
                vg(alpha, iq) = 0.;
            }
            else {
                vg(alpha, iq) =
                    std::real(phi.col(alpha).dot(dDdk * phi.col(alpha))) / 2. /
                    omega(alpha, iq);
            }
        }
    }
}


Eigen::MatrixXcd decimation(const Eigen::Ref<const Eigen::MatrixXcd>& winp,
                            const Eigen::Ref<const Eigen::MatrixXcd>& tainp,
                            double tolerance,
                            int maxiter)
{
    Eigen::MatrixXcd w(winp);
    Eigen::MatrixXcd ta(tainp);
    Eigen::MatrixXcd tb(ta.adjoint());
    Eigen::MatrixXcd ws(w);
    Eigen::MatrixXcd old(w);

    std::size_t n = static_cast<std::size_t>(tainp.rows());
    Eigen::MatrixXcd axta(n, n);
    Eigen::MatrixXcd axtb(n, n);
    for (int iter = 0; iter < maxiter; ++iter) {
        auto ax = w.fullPivHouseholderQr();
        axta = ax.solve(ta);
        axtb = ax.solve(tb);
        ws -= ta * axtb;
        if ((ws - old).norm() / ws.norm() < tolerance) {
            return ws;
        }
        w -= ta * axtb + tb * axta;
        ta = -ta * axta;
        tb = -tb * axtb;
        old = ws;
    }
    throw Decimation_MaxIterException();
}


void decimation_bridge(const std::complex<double>* k00in,
                       const std::complex<double>* k01in,
                       int ndof,
                       std::complex<double> en,
                       double tolerance,
                       int maxiter,
                       std::complex<double>* inv_g00out)
{
    Eigen::Map<const Eigen::MatrixXcd> k00(k00in, ndof, ndof);
    Eigen::Map<const Eigen::MatrixXcd> k01(k01in, ndof, ndof);
    Eigen::Map<Eigen::MatrixXcd> nruter(inv_g00out, ndof, ndof);

    Eigen::MatrixXcd one(Eigen::MatrixXcd::Identity(ndof, ndof));
    Eigen::MatrixXcd almosthh = k00 - en * one;
    nruter = decimation(almosthh, k01, tolerance, maxiter);
}