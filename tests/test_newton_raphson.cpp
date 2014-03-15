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
#include <cppopt/newton_raphson.h>

namespace co = cppopt;

TEST_CASE("Univariate Newton Raphson Root Finding") {
    cppopt::Matrix x(1, 1);
    
    // Start at x = -0.5, should yield x = 0
    x(0) = co::Scalar(-0.5);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateSample::getFunction(), UnivariateSample::getDerivative(), x) == co::SUCCESS);
    REQUIRE( fabs(x(0) - 0.0) < 0.001);
    
}

TEST_CASE("Univariate Newton Raphson Maximum / Minimum Finding") {
    cppopt::Matrix x(1, 1);
    
    // Start at x = -0.5, should yield x = 0 (Minimum)
    x(0) = co::Scalar(-0.5);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateSample::getDerivative(), UnivariateSample::getSecondDerivative(), x) == co::SUCCESS);
    REQUIRE( fabs(x(0) - 0.0) < 0.001);
    
    // Start at x = -0.7, should yield x = -2.8024 (Maximum)
    x(0) = co::Scalar(-0.7);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateSample::getDerivative(), UnivariateSample::getSecondDerivative(), x) == co::SUCCESS);
    REQUIRE( fabs(x(0) - -2.8024) < 0.001);
    
    // Start at x = 2, should yield 2.17080
    x(0) = co::Scalar(2);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateSample::getDerivative(), UnivariateSample::getSecondDerivative(), x) == co::SUCCESS);
    REQUIRE( fabs(x(0) - 2.17080) < 0.01);
}