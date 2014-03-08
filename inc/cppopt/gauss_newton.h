// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.


#ifndef CPPOPT_GAUSS_NEWTON
#define CPPOPT_GAUSS_NEWTON

#include "types.h"

namespace cppopt {
    
    /** Performs one step of non-linear least squares optimization using the Gauss-Newton algorithm.
     *
     *  The Gauss-Newton method is an approximation to the Newton method for the special case of 
     *  non-linear least squares. Newton's method, in case of optimization, is given by
     *
     *  x_n+1 = x - df/dx / ddf/dx
     *
     *  where df/dx is the first derivative of f w.r.t. to x and ddf/dx is the second derivative.
     *  Besides the fact that calculating the second derivative is computationally expensive, in
     *  the case of a multivariate function ddf/dx (also called the Hessian matrix) becomes highly
     *  dimensional.
     *
     *  The Gauss-Newton avoids direct calculation of the Hessian matrix and instead approximates
     *  the Hessian matrix.
     *
     *  \param f Function object evaluating the residuals at x.
     *           Input: variables given as vector of size Nx1.
     *           Output: function values given as vector of size Mx1.
     *  \param d First order partial derivatives of f at x.
     *           Input: variables given as vector of size Nx1.
     *           Output: all partial derivates of f at x given as matrix of size MxN.
     *  \param x Variables of function f given as vector of size Nx1 vector. Will be updated in case of success.
     *  \return Status indicating success or failure due to ill-conditioned input.
     */
    ResultInfo gaussNewton(const F &f, const F &d, Matrix &x) {
        Matrix j = d(x);
        
        // Make sure we have more functions than variables.
        assert(j.rows() >= j.cols());
        
        Matrix jt = j.transpose();
        Matrix y = f(x);

        auto llt = (jt * j).ldlt();
        if (llt.info() != Eigen::Success) {
            return ERROR;
        }

        Matrix s = llt.solve(jt * y * Scalar(-1));        
        x = x + s;

        return SUCCESS;
    }
}

#endif