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

/*
 This example optimizes f(x) = 3x^3 - 10x^2 - 56x + 5 with start value x = 2.
 
 Note that
    f'(x)  = 9x^2 - 20x - 56
    f''(x) = 18x - 20
 */

// x^2 + y^2 + 2x + 8y
// df(x,y)/dx = 2x + 2
// df(x,y)/dy = 2y + 8
//
// ddf/ddx = 2
// ddf/dxdy = 0
// ddf/dydx = 0
// ddf/ddy = 2



int main() {
    
    

    cppopt::F df = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(2, 1);
        
        d(0) = 2.f * x(0) + 2.f;
        d(1) = 2.f * x(1) + 8.f;
        
        return d;
    };
    
    cppopt::F ddf = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(2, 2);
        
        d(0,0) = 2.f;
        d(0,1) = 0.f;
        d(1,0) = 0.f;
        d(1,1) = 2.f;
        
        return d;
    };
    
    cppopt::Matrix x(2, 1);
    x(0) = -3.f;
    x(1) = -2.f;
    
    // Iterate while norm of residual is greater than a user-selected threshold.
    while (df(x).norm() > 0.001f) {
        cppopt::newtonRaphson(df, ddf, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << df(x).norm() << std::endl;
    }
    
    //std::cout << "Found a " << (ddf(x)(0) < 0.f ? "Maximum" : "Minimum") << std::endl;
    
    return 0;
}