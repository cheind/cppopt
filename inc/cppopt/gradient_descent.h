// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.


#ifndef CPPOPT_GRADIENT_DESCENT
#define CPPOPT_GRADIENT_DESCENT

#include "types.h"

namespace cppopt {
    
    /** Performs one step of the numerical minimization using the method of steepest descent.
     *
     *  Assume f is real-valued possibly multivariate function of x and df/dx exists. One can 
     *  observe that f decreases fastest in the direction of the negative gradient. A step length
     *  parameter, that is allowed to change in every iteration, determines the magnitude of the 
     *  step taken.
     *
     *  \param d First order partial derivatives of f at x.
     *           Input: variables given as vector of size Nx1.
     *           Output: all partial derivates of f at x given as row vector 1xN.
     *  \param x Variables of function f given as vector of size Nx1 vector. Will be updated in case of success.
     *  \return Status indicating success or failure due to ill-conditioned input.
     */
    ResultInfo gradientDescent(const F &d, Matrix &x, Scalar step) {
        x = x - step * d(x).transpose();
        return SUCCESS;
    }
}
			
#endif