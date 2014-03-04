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
    AffineTransform2D torig = Eigen::Translation2f(20.f, -10.f) * Eigen::Rotation2Df(0.4f);

    cppopt::Matrix sp(20, 2);
    for (int i = 0; i < mp.rows(); ++i) {
        sp.row(i) = torig * Eigen::Vector2f(mp.row(i));
    }

    // Define our residual function
    cppopt::F f = [&mp, &sp](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix y(mp.rows(), 1);

        AffineTransform2D t = Eigen::Translation2f(x(1), x(2)) * Eigen::Rotation2Df(x(0));
        
        for (int i = 0; i < y.rows(); ++i) {
            Eigen::Vector2f m = mp.row(i);
            Eigen::Vector2f s = sp.row(i);
            y(i) = (m - t * s).norm();
        }
        
        return y;
    };
    
    // Define the Jacobian of the residual function
    cppopt::F df = [&mp, &sp](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(mp.rows(), x.rows());

        AffineTransform2D t = Eigen::Translation2f(x(1), x(2)) * Eigen::Rotation2Df(x(0));
        
        AffineTransform2D td_dth;
        td_dth.matrix() << -sinf(x(0)), -cosf(x(0)), 0
                         , cosf(x(0)), -sinf(x(0)), 0
                         , 0, 0, 1;

        AffineTransform2D td_dtx;
        td_dtx.matrix() << 0, 0, 1
                         , 0, 0, 0
                         , 0, 0, 1;

        AffineTransform2D td_dty;
        td_dty.matrix() << 0, 0, 0
                         , 0, 0, 1
                         , 0, 0, 1;
        
        for (int i = 0; i < d.rows(); ++i) {
            Eigen::Vector2f m = mp.row(i);
            Eigen::Vector2f s = sp.row(i);
            Eigen::Vector2f st = t * s;
            Eigen::Vector2f sdt_dth = td_dth * st;
            Eigen::Vector2f sdt_dtx = td_dtx * st;
            Eigen::Vector2f sdt_dty = td_dty * st;

            cppopt::Scalar denom = 2.0f * (m - st).norm();
            d(i, 0) = (-2.f * (m(0) - st(0)) * sdt_dth(0) - 2.f * (m(1) - st(1)) * sdt_dth(1)) / denom;
            d(i, 1) = (-2.f * (m(0) - st(0)) * sdt_dtx(0) - 2.f * (m(1) - st(1)) * sdt_dtx(1)) / denom; 
            d(i, 2) = (-2.f * (m(0) - st(0)) * sdt_dty(0) - 2.f * (m(1) - st(1)) * sdt_dty(1)) / denom;
        }
        
        
        return d;
    };

    // Create start solution
    cppopt::Matrix x(3, 1);
    x << 0.0f, 0.f, 0.0f;
    
    // Iterate while norm of residual is greater than a user-selected threshold.
    while (f(x).norm() > 0.001f) {
        cppopt::gaussNewton(f, df, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << f(x).norm() << std::endl;
    }

    AffineTransform2D tfinal = Eigen::Translation2f(x(1), x(2)) * Eigen::Rotation2Df(x(0));
    // We are actually searching for the inverse of torig i.e the matrix that brings back the 
    // transformed scenepoints into alignment with the model points.
    std::cout << (torig * tfinal).matrix() << std::endl;
    return 0;
    
}