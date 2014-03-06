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
#include <Eigen/Geometry>
#include <random>
#include <iostream>
#include <iomanip>

int main() {
    
    // Generate random points in two dimensions. Let's call those points the model points.
    cppopt::Matrix mp = cppopt::Matrix::Random(20, 2) * 100.f;
    
    // The scene points will be a translated and rotated version of the model points plus some optional noise.
    typedef Eigen::Transform<cppopt::Scalar, 2, Eigen::Affine> AffineTransform2D;
    AffineTransform2D torig;
    torig.setIdentity();
    torig.rotate(0.3f);
    
    cppopt::Matrix sp(20, 2);
    for (int i = 0; i < mp.rows(); ++i) {
        sp.row(i) = torig * Eigen::Vector2f(mp.row(i));
    }

    // Define our residual function
    cppopt::F f = [&mp, &sp](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix y(mp.rows(), 1);

        AffineTransform2D t;
        t.setIdentity();
        t.rotate(x(0));

        for (int i = 0; i < y.rows(); ++i) {
            Eigen::Vector2f m = mp.row(i);
            Eigen::Vector2f s = sp.row(i);
            y(i) = 0 - (m - t * s).norm();
        }
       
        return y;
    };
    
    // Define the Jacobian of the residual function
    cppopt::F df = [&mp, &sp](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(mp.rows(), x.rows());
       
        AffineTransform2D t;
        t.setIdentity();
        t.rotate(x(0));
        
        for (int i = 0; i < d.rows(); ++i) {
            Eigen::Vector2f m = mp.row(i);
            Eigen::Vector2f s = sp.row(i);

            cppopt::Scalar denom = -2.f * (m - t * s).norm();
            
            d(i, 0) = (2.f * (m(0) - s(0)*cosf(x(0)) + s(1)*sinf(x(0))) * (s(0)*sinf(x(0)) + s(1)*cosf(x(0)))) +
                      (2.f * (m(1) - s(0)*sinf(x(0)) - s(1)*cosf(x(0))) * (-s(0)*cosf(x(0)) + s(1)*sinf(x(0))));
            d(i, 0) /= denom;
            
        }        
        
        return d;
    };

    // Create start solution
    cppopt::Matrix x(1, 1);
    x << 0.0f;
    
    // Iterate while norm of residual is greater than a user-selected threshold.
    cppopt::ResultInfo ri = cppopt::SUCCESS;
    while (ri == cppopt::SUCCESS && f(x).norm() > 0.0001f) {
        ri = cppopt::gaussNewton(f, df, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << f(x).norm() << std::endl;
    }

    if (ri == cppopt::SUCCESS) {
        AffineTransform2D tfinal;
        tfinal.setIdentity();
        tfinal.rotate(x(0));

        std::cout << "The following matrix should be close to identity." << std::endl;
        std::cout << (torig * tfinal).matrix() << std::endl;
    } else {
        std::cout << "Failed to compute." << std::endl;
    }

    return 0;
    
}