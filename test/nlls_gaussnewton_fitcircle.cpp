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

/** This example fits a circle to a set of two-dimensional points using non-linear least squares.
 *
 *  The algorithm used is the iterative Gauss-Newton for non-linear least squares. This algorithm
 *  requires a function that calculates the residuals (geometric errors) between each data point
 *  and the current estimate for the circle parameters. The geometric error of a point p to a circle 
 *  given by center and radius is defined by
 *
 *      r(p, circle, radius) = radius - sqrt((p.x - center.x)^2 + (p.y - center.y)^2)
 *
 *  Additionally all partial derivates of the residual function are required
 *
 *      dr/dcenter.x = (2 * (p.x - center.x)) / (2 * sqrt((p.x - center.x)^2 + (p.y - center.y)^2))
 *      dr/dcenter.y = (2 * (p.y - center.y)) / (2 * sqrt((p.x - center.x)^2 + (p.y - center.y)^2))
 *      dr/dradius   = 1
 **/

/** Generate random points on circle. 
 *
 * \param center circle center
 * \param radius radius of circle
 * \param sigma sigma parameter of white noise which is added to perfect circle points.
 * \return Matrix of size 20x2 containing generated circle points in rows.
 */
cppopt::Matrix generatePointsOnCircle(const cppopt::Vector &center, cppopt::Scalar radius, cppopt::Scalar sigma) {
    cppopt::Matrix p(20, 2);
    
    std::default_random_engine generator;
    std::uniform_real_distribution<cppopt::Scalar> ud(0, 1);
    std::normal_distribution<cppopt::Scalar> nd(0, sigma);
    
    for (int i = 0; i < p.rows(); ++i) {
        float angle = ud(generator) * cppopt::Scalar(3.1415f * 2.f);
        cppopt::Vector x(2);
        x << cosf(angle) * radius + nd(generator),
             sinf(angle) * radius + nd(generator);
        p.row(i) = x + center;
    }
    
    return p;
}

int main() {
    
    // Generate random points on circle and 
    cppopt::Vector c(2);
    c << 2.f, 1.5f;
    cppopt::Matrix p = generatePointsOnCircle(c, 8.f, 0.001f);
        
    // Define the residual function. Returns a matrix of size Nx1 containing the geometric
    // distances for each circle point to the current circle estimate given in x.
    cppopt::F f = [&p](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix y(p.rows(), 1);
        
        for (int i = 0; i < y.rows(); ++i) {
            y(i) = x(2) - sqrtf(powf(x(0)-p(i, 0), 2.f) + powf(x(1)-p(i, 1), 2.f));
        }
        
        return y;
    };
    
    // Define the Jacobian of the residual function. That is the matrix of size Nx2 that
    // contains all partial derivates.
    cppopt::F df = [&p](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(p.rows(), x.rows());
        
        for (int i = 0; i < d.rows(); ++i) {
            cppopt::Scalar denom = 2 * sqrtf(powf(x(0)-p(i, 0), 2.f) + powf(x(1)-p(i, 1), 2.f));
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
    cppopt::ResultInfo ri = cppopt::SUCCESS;
    while (ri == cppopt::SUCCESS && f(x).norm() > 0.01f) {
        ri = cppopt::gaussNewton(f, df, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << f(x).norm() << std::endl;
    }
    
    return 0;
}