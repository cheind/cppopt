// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef CPPOPT_MULTIVARIATE_FUNCTION
#define CPPOPT_MULTIVARIATE_FUNCTION

#include <cppopt/types.h>

/** Represents the function sin(x) + cos(y) and derivatives. 
 *  See: https://www.wolframalpha.com/input/?i=%28sin%28x%29+%2B+cos%28y%29%29
 */
struct MultivariateTestSample {
    static cppopt::F getFunction() {
        return [](const cppopt::Matrix &x) -> cppopt::Matrix {
            cppopt::Matrix r(1, 1);
            
            r(0) = cppopt::Scalar(sin(x(0))) + cppopt::Scalar(cos(x(1)));
            
            return r;
        };
    }

    static cppopt::F getDerivative() {
        return [](const cppopt::Matrix &x) -> cppopt::Matrix {
            cppopt::Matrix r(2, 1);
            
            r(0) = cppopt::Scalar(cos(x(0)));
            r(1) = cppopt::Scalar(-sin(x(1)));
            
            return r;
        };
    }
    
    static cppopt::F getSecondDerivative() {
        return [](const cppopt::Matrix &x) -> cppopt::Matrix {
            cppopt::Matrix r(2, 2);
            
            r(0,0) = cppopt::Scalar(-sin(x(0)));
            r(0,1) = cppopt::Scalar(0);
            r(1,0) = cppopt::Scalar(0);
            r(1,1) = cppopt::Scalar(-cos(x(1)));
            
            return r;
        };
    }
};

#endif