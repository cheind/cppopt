// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include "catch.hpp"

#include <cppopt/newton_raphson.h>
#include "univariate_function.h"
#include "multivariate_function.h"

namespace co = cppopt;

TEST_CASE("Univariate Newton Raphson Root Finding") {
    cppopt::Matrix x(1, 1);
    
    // Start at x = -0.5, should yield x = 0
    x(0) = co::Scalar(-0.5);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateTestSample::getFunction(), UnivariateTestSample::getDerivative(), x) == co::SUCCESS);
    REQUIRE( fabs(x(0) - 0.0) < 0.001);
    
}

TEST_CASE("Univariate Newton Raphson Maximum / Minimum Finding") {
    cppopt::Matrix x(1, 1);
    
    // Start at x = -0.5, should yield x = 0 (Minimum)
    x(0) = co::Scalar(-0.5);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateTestSample::getDerivative(), UnivariateTestSample::getSecondDerivative(), x) == co::SUCCESS);
    REQUIRE(fabs(x(0) - 0.0) < 0.001);
    
    // Start at x = -0.7, should yield x = -2.8024 (Maximum)
    x(0) = co::Scalar(-0.7);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateTestSample::getDerivative(), UnivariateTestSample::getSecondDerivative(), x) == co::SUCCESS);
    REQUIRE(fabs(x(0) - -2.8024) < 0.001);
    
    // Start at x = 2, should yield 2.17080
    x(0) = co::Scalar(2);
    for (int i = 0; i < 10; ++i)
        REQUIRE( co::newtonRaphson(UnivariateTestSample::getDerivative(), UnivariateTestSample::getSecondDerivative(), x) == co::SUCCESS);
    REQUIRE(fabs(x(0) - 2.17080) < 0.01);
}

TEST_CASE("Newton Raphson Single Step Solutions") {
    
    {
        // By definition of the method the root of a linear function should be found in a single step
        // no matter of the starting condition.
        co::Matrix x(1, 1);
        x(0) = co::Scalar(-20.0);
        
        co::F f = [](const co::Matrix &x) -> co::Matrix {
            co::Matrix r(1, 1);
            r(0) = co::Scalar(2.5) * x(0) - co::Scalar(3);
            return r;
        };
    
        co::F d = [](const co::Matrix &x) -> co::Matrix {
            co::Matrix r(1, 1);
            r(0) = co::Scalar(2.5);
            return r;
        };
        
        REQUIRE(co::newtonRaphson(f, d, x) == co::SUCCESS);
        REQUIRE(fabs(x(0) - 1.2) < 0.0001);
    }
    
    {
        // By definition of the method the minimum of a quadric function should be found in a single step,
        // independent of the starting position
        co::Matrix x(1, 1);
        x(0) = co::Scalar(-20);
        
        co::F f = [](const co::Matrix &x) -> co::Matrix {
            co::Matrix r(1, 1);
            r(0) = co::Scalar(2) * x(0) * x(0) - co::Scalar(3) * x(0) - co::Scalar(5);
            return r;
        };
        
        co::F d = [](const co::Matrix &x) -> co::Matrix {
            co::Matrix r(1, 1);
            r(0) = co::Scalar(4) * x(0) - co::Scalar(3);
            return r;
        };
        
        co::F dd = [](const co::Matrix &x) -> co::Matrix {
            co::Matrix r(1, 1);
            r(0) = co::Scalar(4);
            return r;
        };
        
        REQUIRE(co::newtonRaphson(d, dd, x) == co::SUCCESS);
        REQUIRE(fabs(x(0) - 0.75) < 0.0001);
        REQUIRE(fabs(f(x)(0) - -6.125) < 0.0001);
    }
    
    {
        // By definition of the method the minimum of a multivariate quadric function should be found in a single step,
        // independent of the start position
        
        co::Matrix x(2, 1);
        x(0) = co::Scalar(-20);
        x(1) = co::Scalar(-20);
        
        co::F d = [](const cppopt::Matrix &x) -> cppopt::Matrix {
            co::Matrix d(2, 1);
            
            d(0) = co::Scalar(2) * x(0) + co::Scalar(2);
            d(1) = co::Scalar(2) * x(1) + co::Scalar(8);
            
            return d;
        };
        
        co::F dd = [](const cppopt::Matrix &x) -> cppopt::Matrix {
            co::Matrix d(2, 2);
            
            d(0,0) = co::Scalar(2);
            d(0,1) = co::Scalar(0);
            d(1,0) = co::Scalar(0);
            d(1,1) = co::Scalar(2);
            
            return d;
        };
        
        REQUIRE(co::newtonRaphson(d, dd, x) == co::SUCCESS);
        REQUIRE(fabs(x(0) - -1) < 0.0001);
        REQUIRE(fabs(x(1) - -4) < 0.0001);
    }
}

TEST_CASE("Multivariate Newton Raphson Maximum / Minimum Finding") {
    
    cppopt::Matrix x(2, 1);
    
    // Start at (1.3,-0.1) should converge to nearest maximum at (pi/2, 0)
    x(0) = co::Scalar(1.3);
    x(1) = co::Scalar(-0.1);
    for (int i = 0; i < 10; ++i)
        REQUIRE(co::newtonRaphson(MultivariateTestSample::getDerivative(), MultivariateTestSample::getSecondDerivative(), x) == co::SUCCESS);
    
    REQUIRE(fabs(x(0) - 3.141592/2) < 0.001);
    REQUIRE(fabs(x(1) - 0) < 0.001);
    
    // Start at (-2,3) should converge to nearest minimum at (-pi/2, pi)
    x(0) = co::Scalar(-2);
    x(1) = co::Scalar(3);
    for (int i = 0; i < 10; ++i)
        REQUIRE(co::newtonRaphson(MultivariateTestSample::getDerivative(), MultivariateTestSample::getSecondDerivative(), x) == co::SUCCESS);
    
    REQUIRE(fabs(x(0) - -3.141592/2) < 0.001);
    REQUIRE(fabs(x(1) - 3.141592) < 0.001);
    
    
    // Start at (0,0) should fail, because the Jacobian will become degenerate. Should this be handled in some way?
    x(0) = co::Scalar(0);
    x(1) = co::Scalar(0);
    REQUIRE(co::newtonRaphson(MultivariateTestSample::getDerivative(), MultivariateTestSample::getSecondDerivative(), x) == co::ERROR);
}


