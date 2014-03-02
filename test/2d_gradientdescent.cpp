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

/*
 This example optimizes f(x,y) = x^2 + y^2 + 2x + 8y which has a global minimum at (-1, -4).
 
 First order (Gradient)
 df/dx = 2x + 2
 df/dy = 2y + 8
 
 */

//http://pages.cs.wisc.edu/~ferris/cs730/chap3.pdf

int main() {
    
    cppopt::F df = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(2, 1);
        
        d(0) = 2.f * x(0) + 2.f;
        d(1) = 2.f * x(1) + 8.f;
        
        return d;
    };
    
    cppopt::Matrix x(2, 1);
    x(0) = -3.f;
    x(1) = -2.f;
    
    // Iterate while norm of residual is greater than a user-selected threshold.
    while (df(x).norm() > 0.001f) {
        cppopt::gradientDescent(df, x, 0.001f);
        std::cout
        << std::fixed << std::setw(3)
        << "Parameters: " << x.transpose()
        << " Error: " << df(x).norm() << std::endl;
    }
    
    return 0;
}