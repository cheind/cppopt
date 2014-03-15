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

/** This example minimizes the squared error between two point sets by determining the rotation that best aligns them.
 *
 *  The algorithm used is the iterative Gauss-Newton for non-linear least squares. This algorithm
 *  requires a function that calculates the residuals (geometric errors) between the data points of both 
 *  point sets. For this example we assume that each point in the first point set has a matching point in the second
 *  point set and those points are identified by having the same array index. The geometric error for an individual 
 *  point pair, m and s - rotated in plane by angle phi, is given by 
 *
 *      r(m, s, phi) = ||m - R(phi) * s|| with
 *      R(phi) = |cos(phi) -sin(phi)|
 *               |sin(phi)  cos(phi)|
 *
 *  Additionally all partial derivates of the residual function are required. Setting the column vector k to
 *  
 *      k(phi) = m - R(phi) * s
 *
 *  the residual function can be rewritten as
 *
 *      r(m, s, phi) = sqrt(k' * k)
 * 
 *  and thus the required partial derivatives can be written as
 *
 *      dr/dphi = (k' * dk/dphi + dk'/dphi * k) / (2 * sqrt(k' * k))
 *
 *  where 
 *      
 *      dk/dphi = -(dR(phi)/dphi * si)
 *      dk'/dphi = -(dR(phi)/dphi * si)'
 *
 *  and 
 *
 *      dR/dphi = |-sin(phi) -cos(phi)|
 *                | cos(phi) -sin(phi)|
 *
 **/

/** Any affine transform in 2D */
typedef Eigen::Transform<cppopt::Scalar, 2, Eigen::Affine> AffineTransform2D;

/** Vector of two dimensions */
typedef Eigen::Matrix<cppopt::Scalar, 2, 1> Vector2D;

/** Generate two-dimensional point sets to be aligned via a rotation.
 *
 * \param pointsModel points of model.
 * \param pointsScene points of scene.
 * \param rotation Angle of rotation in radians.
 * \param sigma Sigma parameter of white noise which is to points of scene before rotation.
 */
void generatePointSets(cppopt::Matrix &pointsModel, cppopt::Matrix &pointsScene, cppopt::Scalar rotation, cppopt::Scalar sigma) {
    
    // Model points are simply randomly generated
    pointsModel = cppopt::Matrix::Random(20, 2) * 100.f;
    
    // The scene points will be a translated and rotated version of the model points plus some optional noise.
    AffineTransform2D torig;
    torig.setIdentity();
    torig.rotate(rotation);
    
    std::default_random_engine generator;
    std::normal_distribution<cppopt::Scalar> nd(0, sigma);
    
    pointsScene.resize(pointsModel.rows(), 2);
    for (int i = 0; i < pointsModel.rows(); ++i) {
        Vector2D noise(nd(generator), nd(generator));
        pointsScene.row(i) = torig * (Vector2D(pointsModel.row(i)) + noise);
    }
}

int main() {
    
    // Generate random points in two dimensions.
    cppopt::Matrix mp, sp;
    generatePointSets(mp, sp, 0.3f, 0.001f);

    // Define the residual function that measures the geometric error between the two point sets with respect to the
    // current rotation estimate. Note that it is assumed that the points within sets are linked via indices.
    cppopt::F f = [&mp, &sp](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix y(mp.rows(), 1);

        AffineTransform2D t;
        t.setIdentity();
        t.rotate(x(0));
        
        for (int i = 0; i < y.rows(); ++i) {
            Vector2D m = mp.row(i);
            Vector2D s = sp.row(i);
            y(i) = (m - t * s).norm();
        }
       
        return y;
    };
    
    // Define the Jacobian of the residual function.
    cppopt::F df = [&mp, &sp](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(mp.rows(), x.rows());
       
        AffineTransform2D t;
        t.setIdentity();
        t.rotate(x(0));

        AffineTransform2D dt;
        dt.matrix() << -sinf(x(0)), -cosf(x(0)), 0,
                        cosf(x(0)), -sinf(x(0)), 0,
                        0, 0, 1;
        
        for (int i = 0; i < d.rows(); ++i) {
            Vector2D m = mp.row(i);
            Vector2D s = sp.row(i);
            Vector2D k = m - t * s;
            Vector2D kd = dt * s * -1.f;

            cppopt::Scalar nom = (k.transpose() * kd + kd.transpose() * k)(0);
            cppopt::Scalar denom = 2.f * k.norm();
            
            d(i, 0) = nom / denom;            
        }        
        
        return d;
    };

    // Create start solution
    cppopt::Matrix x(1, 1);
    x << 0.0f;
    
    // Iterate while norm of residual is greater than a user-selected threshold.
    cppopt::ResultInfo ri = cppopt::SUCCESS;
    while (ri == cppopt::SUCCESS && f(x).norm() > 0.01f) {
        ri = cppopt::gaussNewton(f, df, x);
        std::cout
            << std::fixed << std::setw(3)
            << "Parameters: " << x.transpose()
            << " Error: " << f(x).norm() << std::endl;
    }
    
    assert(fabs(x(0) - cppopt::Scalar(-0.3)) < cppopt::Scalar(0.001));

    return 0;
    
}