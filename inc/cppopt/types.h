// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef CPPOPT_TYPES
#define CPPOPT_TYPES

#include <Eigen/Dense>
#include <functional>

namespace cppopt {
    
    // Default precision
    typedef float Scalar;
    
    // Default matrix type
    typedef Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic > Matrix;
    
    // Default vector type
    typedef Eigen::Matrix< Scalar, Eigen::Dynamic, 1 > Vector;
    
    // Function prototype
    typedef std::function< Matrix(const Matrix &x) > F;
}

#endif
