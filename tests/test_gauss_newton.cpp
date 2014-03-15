// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include "catch.hpp"

#include <cppopt/gauss_newton.h>

namespace co = cppopt;

TEST_CASE("Gauss Newton non-linear least squares.") {
    
    // This is based on the example given at http://en.wikipedia.org/wiki/Gauss–Newton_algorithm
    
    cppopt::Matrix samples(2, 7);
    samples <<  co::S(0.038), co::S(0.194), co::S(0.425), co::S(0.626), co::S(1.253), co::S(2.500), co::S(3.740),      // S
                co::S(0.050), co::S(0.127), co::S(0.094), co::S(0.2122), co::S(0.2729), co::S(0.2665), co::S(0.3317);  // Rate
    
    // Residual function
    cppopt::F f = [&samples](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix y(samples.cols(), 1);
        
        for (int i = 0; i < y.rows(); ++i) {
            y(i) = samples(1, i) - (x(0) * samples(0, i)) / (x(1) + samples(0,i));
        }
        
        return y;
    };
    
    // Jacobian of residual
    cppopt::F df = [&samples](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(samples.cols(), x.rows());
        
        for (int i = 0; i < d.rows(); ++i) {
            co::Scalar denom = (x(1) + samples(0, i)) * (x(1) + samples(0, i));
            d(i, 0) = -samples(0, i) / (x(1) + samples(0, i));
            d(i, 1) = (x(1) * samples(0, i)) / denom;
        }
        
        
        return d;
    };
    
    // Create start solution
    cppopt::Matrix x(2, 1);
    x << co::S(0.9), co::S(0.2);
    
    // Sum of squared residuals at beginning
    REQUIRE(fabs((f(x).transpose() * f(x))(0) - 1.445) < 0.01);
    
    for (int i = 0; i < 5; ++i)
        REQUIRE(co::gaussNewton(f, df, x) == co::SUCCESS);
    
    // Sum of squared residuals after optimization
    REQUIRE(fabs((f(x).transpose() * f(x))(0) - 0.00784) < 0.0001);
    
    // Check final parameters
    REQUIRE(fabs(x(0) - 0.362) < 0.01);
    REQUIRE(fabs(x(1) - 0.556) < 0.01);
    
    
    
}