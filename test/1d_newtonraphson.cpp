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
 


int main() {

    cppopt::F df = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(1, 1);
        
        d(0) = 9.f * powf(x(0), 2) - 20.f * x(0) - 56.f;
        
        return d;
    };
    
    cppopt::F ddf = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(1, 1);
        
        d(0) = 18 * x(0) - 20.f;
        
        return d;
    };
    
    cppopt::Matrix x(1, 1);
    x(0) = 2.f; // Try to start with 0 and you fill find a maximum.
    
    // Iterate while norm of residual is greater than a user-selected threshold.
    while (df(x).norm() > 0.001f) {
        cppopt::newtonRaphson(df, ddf, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << df(x).norm() << std::endl;
    }
    
    std::cout << "Found a " << (ddf(x)(0) < 0.f ? "Maximum" : "Minimum") << std::endl;
    
    return 0;
}