// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef CPPOPT_FINITE_DIFFERENCE
#define CPPOPT_FINITE_DIFFERENCE

#include "types.h"
#include <limits>

namespace cppopt {
    
    namespace internal {
        
        /** Helper method to find a suitable H value that determines the step size in numerical differentation.
         *
         *  The returned value depends on the datatype and  value of the variable it is calculated for. Caution is taken 
         *  to generate a number that
         *   - small
         *   - machine representable
         *   - for which x + h is also machine representable
         *
         * For this reasons the implementation of this method is rather slow.
         *
         * \param x value to generate h for
         * \return h
         */
        template<class T>
        T findSuitableH(T x) {
            
            // Note this dance is only necessary due to numerical rounding issues.
            // See http://en.wikipedia.org/wiki/Numerical_differentiation for details.
            
            const T eps = static_cast<T>(sqrt(std::numeric_limits<T>::epsilon()));
            const T h = eps * x;
            volatile T xph = x + h;
            return xph - x;
            
        }  
    }
    
    /** Approximates the nth order derivative for a multivariate scalar valued function using forward difference. */
    template<int Order>
    Matrix forward_difference(const F & f, const Dims &dims, const Matrix &x);
    
    /** Approximates the first order derivative for a multivariate scalar valued function using forward difference. */
    template<>
    Matrix forward_difference<1>(const F & f, const Dims &dims, const Matrix &x) {
        Matrix d(dims.y_rows, dims.x_rows);
        Matrix h(dims.x_rows, 1);
    
        for (int i = 0; i < x.rows(); ++i) {
            h.setZero();
            
            const Scalar dx = internal::findSuitableH(x(i));
            h(i, 0) = dx;
            d.col(i) = (f(x + h) - f(x)) / dx;
        }
        
        return d;
    }
}

#endif