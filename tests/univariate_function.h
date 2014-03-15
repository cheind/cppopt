// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef CPPOPT_UNIVARIATE_FUNCTION
#define CPPOPT_UNIVARIATE_FUNCTION

#include <cppopt/types.h>

/** Represents the function sin(x^2) and its first order derivative. */
struct UnivariateSample {
    static cppopt::F getFunction() {
        return [](const cppopt::Matrix &x) -> cppopt::Matrix {
            cppopt::Matrix r(1, 1);
            r(0) = cppopt::Scalar(sin(x(0)*x(0)));
            return r;
        };
    }

    static cppopt::F getDerivative() {
        return [](const cppopt::Matrix &x) -> cppopt::Matrix {
            cppopt::Matrix r(1, 1);
            r(0) = 2 * x(0) * cppopt::Scalar(cos(x(0)*x(0)));
            return r;
        };
    }
    
    static cppopt::F getSecondDerivative() {
        return [](const cppopt::Matrix &x) -> cppopt::Matrix {
            cppopt::Matrix r(1, 1);
            r(0) = 2 * (cos(x(0)*x(0)) - 2 * x(0)*x(0)*sin(x(0)*x(0)));
            return r;
        };
    }
};

#endif