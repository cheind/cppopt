// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include "catch.hpp"

#include "univariate_function.h"
#include <cppopt/gradient_descent.h>

namespace co = cppopt;

TEST_CASE("Univariate gradient descent") {
    cppopt::Matrix x(1, 1);
    
    // Start at x = -0.5, should yield x = 0
    x(0) = co::Scalar(-0.5);
    for (int i = 0; i < 20; ++i)
        REQUIRE( co::gradientDescent(UnivariateSample::getDerivative(), x, co::Scalar(0.1)) == co::SUCCESS);
    REQUIRE( fabs(x(0) - 0.0) < 0.01);
    
    // Start at x = 2, should yield
    x(0) = co::Scalar(2);
    for (int i = 0; i < 20; ++i)
        REQUIRE( co::gradientDescent(UnivariateSample::getDerivative(), x, co::Scalar(0.1)) == co::SUCCESS);
    REQUIRE( fabs(x(0) - 2.17080) < 0.01);
    
}
