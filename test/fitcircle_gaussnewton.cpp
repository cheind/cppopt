// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#include <cppopt/gauss_newton.h>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <iomanip>

cppopt::Matrix generatePointsOnCircle(const cppopt::Vector &center, cppopt::Scalar radius, cppopt::Scalar sigma) {
    cppopt::Matrix p(20, 2);
    
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    
    for (int i = 0; i < p.rows(); ++i) {
        float angle = distribution(generator) * cppopt::Scalar(3.1415f * 2.f);
        cppopt::Vector x(2);
        x << cosf(angle) * radius, sinf(angle) * radius;
        p.row(i) = x + center;
    }
    
    return p;
}

int main() {
    
    cppopt::Vector c(2);
    c << 2.f, 1.5f;
    cppopt::Matrix p = generatePointsOnCircle(c, 8.f, 0);
        
        
    cppopt::F f = [&p](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix y(p.rows(), 1);
        
        for (int i = 0; i < y.rows(); ++i) {
            y(i) = x(2) - sqrtf(powf(x(0)-p(i, 0), 2.f) + powf(x(1)-p(i, 1), 2.f));
        }
        
        return y;
    };
    
    cppopt::F df = [&p](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(p.rows(), x.rows());
        
        for (int i = 0; i < d.rows(); ++i) {
            typename cppopt::Scalar denom = 2 * sqrtf(powf(x(0)-p(i, 0), 2.f) + powf(x(1)-p(i, 1), 2.f));
            d(i, 0) = (2.f*(p(i, 0) - x(0))) / denom;
            d(i, 1) = (2.f*(p(i, 1) - x(1))) / denom;
            d(i, 2) = 1;
        }
        
        
        return d;
    };
    
    // Create start solution
    cppopt::Matrix x(3, 1);
    x << 2.f, 2.5f, 10.f;
    
    // Iterate while norm of residual is greater than a user-selected threshold.
    while (f(x).norm() > 0.001f) {
        cppopt::gaussNewton(f, df, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << f(x).norm() << std::endl;
    }
    
    return 0;
}

/*
 {
 typedef InputOutputTraits<float, 1, 1, 1, 1> IOT;
 typedef Function<float, IOT> F;
 typedef Function<float, IOT> DF;
 
 F f([](const F::X &x) -> F::Y {typename F::Y y(1); y << sinf(x(0)); return y;});
 DF d([](const DF::X &x) -> DF::Y {typename DF::Y y(1); y << cosf(x(0)); return y;});
 
 typename F::X x(1);
 x << 3.f;
 
 newton(f, d, x);
 std::cout << x << " " << f(x) << std::endl;
 newton(f, d, x);
 std::cout << x << " " << f(x) << std::endl;
 }*/