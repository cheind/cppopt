// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef CPPOPT_NEWTON_RAPHSON
#define CPPOPT_NEWTON_RAPHSON

#include "types.h"

namespace cppopt {
    
    /** Performs one step of the Newton-Raphson root finding algorithm.
     *
     *  The Newton-Raphson algorithm is generally used to iteratively find a root of 
     *  a real-valued univariate function f. In this form it can be shown that a better
     *  position x to the root of f is given by the intersection of the linearization 
     *  of f in x and the x-axis. This leads to to the following iterative algorithm
     *
     *      x_k+1 = x_k - f(x_k) / f'(x_k)
     *  
     *  In case f is vector-valued or multivariate the above formulation is not
     *  directly usable. Instead by rearranging terms this algorithm uses the following
     *  formulation
     *
     *      f'(x_k)(x_k+1 - x_k) = -f(x_k)
     *
     *  In this case f' is an MxN (M..number of functions, N..number of variables) matrix
     *  of all partial derivates, the so called Jacobian matrix. Instead of calculating
     *  inverse of f'(x_k), we set s to (x_k+1 - x_k) and find the solution to s by solving
     *  the linear system of equations f'(x_k)*s = -f(x_k). x_k+1 can then be calculated by
     *
     *      x_k+1 = x_k + s
     *
     *  For the purpose of optimization we observe that root finding of the first order derivative
     *  of the objective function f yields a stationary point of f. A stationary point is defined
     *  by having a gradient of zero. By definition a stationary point can be a maximum, minimum or
     *  saddle point. Therefore there is no guarantee that the algorithm will converge to a minimum.
     *
     *  \param f Function object evaluating f at x. 
     *           Input: variables given as vector of size Nx1.
     *           Output: function values given as vector of size Mx1.
     *  \param d First order partial derivatives of f at x.
     *           Input: variables given as vector of size Nx1.
     *           Output: all partial derivates of f at x given as matrix of size MxN.
     *  \param x Variables of function f given as vector of size Nx1 vector. Will be updated in case of success.
     *  \return Status indicating success or failure due to ill-conditioned input.
     */
    ResultInfo newtonRaphson(const F &f, const F &d, Matrix &x) {
        Matrix jacobian = d(x);
        
        if (jacobian.rows() != jacobian.cols()) {
            // We do cope with more functions than variables.
            return ERROR;
        }
        
        auto lu = jacobian.fullPivLu();
        if (lu.rank() != jacobian.cols()) {
            return ERROR;
        }
        
        Matrix y = f(x);
        Matrix s = lu.solve(y * Scalar(-1));
        x = x + s;
        
        return SUCCESS;
    }
}

#endif