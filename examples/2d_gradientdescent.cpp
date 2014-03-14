// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include <cppopt/gradient_descent.h>
#include <iostream>
#include <iomanip>

/** This example finds a local extremum of a second order multivariate polynomial using the gradient descent algorithm.
 *
 *  The function to be optimized is given by
 *
 *      f(x,y) = x^2 + y^2 + 2x + 8y
 *
 *  which has a global minimum at (-1, -4). The required first order gradient is given by 
 *  Newton-Raphson are given by
 *
 *      df/dx = 2x + 2
 *      df/dy = 2y + 8
 *
 *  This example currently uses a constant for each iteration and does not employ and line-searching techniques. Compare
 *  the results of this example with those of 2d_newtonraphson in terms of number of iterations and accuracy.
 */

int main() {
    
    // Gradient of polynom
    cppopt::F df = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(2, 1);
        
        d(0) = 2.f * x(0) + 2.f;
        d(1) = 2.f * x(1) + 8.f;
        
        return d;
    };
    
    // Start solution
    cppopt::Matrix x(2, 1);
    x(0) = -3.f;
    x(1) = -2.f;
    
    // Iterate while norm of the first order derivative is greater than some predefined threshold.
    cppopt::ResultInfo ri = cppopt::SUCCESS;
    while (ri == cppopt::SUCCESS && df(x).norm() > 0.001f) {
        ri = cppopt::gradientDescent(df, x, 0.01f);
        std::cout
        << std::fixed << std::setw(3)
        << "Parameters: " << x.transpose()
        << " Error: " << df(x).norm() << std::endl;
    }
    
    assert(fabs(x(0) - cppopt::Scalar(-1)) < cppopt::Scalar(0.001));
    assert(fabs(x(1) - cppopt::Scalar(-4)) < cppopt::Scalar(0.001));
    
    return 0;
}