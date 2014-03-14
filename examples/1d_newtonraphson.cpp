// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include <cppopt/newton_raphson.h>
#include <iostream>
#include <iomanip>

/** This example finds a local extremum of a third order univariate polynomial using the Newton-Raphson algorithm.
 *
 *  The function to be optimized is given by 
 *
 *      f(x) = 3x^3 - 10x^2 - 56x + 5
 *
 *  In order to use Newton-Raphson for optimization the first and second order derivates are
 *  required, which are given by
 *
 *      df/dx   = 9x^2 - 20x - 56
 *      df/dxdx = 18x - 20
 *
 *  Note that the Newton-Raphson method is usually used for root finding where one only requires the function and
 *  the first order derivates. Since stationary points (extremum points, saddle points) are defined by having a gradient
 *  of zero (i.e root finding of the first order derivative)
 *
 *      df/dx = 0
 *
 *  we will simply pass the first and second order derivative into Newton-Raphson.
 */

int main() {

    // Define the first order derivative
    cppopt::F df = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(1, 1);
        
        d(0) = 9.f * powf(x(0), 2) - 20.f * x(0) - 56.f;
        
        return d;
    };
    
    // Define the second order derivative
    cppopt::F ddf = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(1, 1);
        
        d(0) = 18 * x(0) - 20.f;
        
        return d;
    };
    
    // Create a start solution. Note that this method does not necessarily find a minimum. Depending on its start value,
    // it will convert to a minimum, maximum or saddle point. For example, try to use zero as start solution and the algorithm
    // will find a maximum instead of a minimum.
    cppopt::Matrix x(1, 1);
    x(0) = 2.f;
    
    // Iterate while norm of the first order derivative is greater than some predefined threshold.
    cppopt::ResultInfo ri = cppopt::SUCCESS;
    while (ri == cppopt::SUCCESS && df(x).norm() > 0.001f) {
        ri = cppopt::newtonRaphson(df, ddf, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << df(x).norm() << std::endl;
    }
    
    std::cout << "Found a " << (ddf(x)(0) < 0.f ? "Maximum" : "Minimum") << std::endl;
    
    assert(fabs(x(0) - cppopt::Scalar(3.841)) < cppopt::Scalar(0.001));
    
    return 0;
}