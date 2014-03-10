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
#include <iostream>
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
            
            //const T eps = static_cast<T>(sqrt(std::numeric_limits<T>::epsilon()));
            /*const T h = 0.00001f;
            //const T h = eps * x;
            volatile T xph = x + h;
            return xph - x;*/

            return 0.001f;
            
        }  
    }
    
    /** Approximates the nth order derivative for a multivariate scalar valued function using forward difference. */
    template<int Order>
    class ApproximateForwardDerivative {};

    /** Numerical approximation of the first order partial derivatives. 
      *
      * This method works for real and vector valued functions that are either
      * univariate or multivariate.
      */
    template<>
    class ApproximateForwardDerivative<1> {
    public:
        
        /** Initialize */
        ApproximateForwardDerivative(const F & f, const Dims &dims) 
            : _f(f), _dims(dims)
        {}

        /** Calculate the first order derivative around x.*/
        Matrix operator()(const Matrix &x) const
        {
            Matrix d(_dims.y_rows, _dims.x_rows);
            Matrix offset(_dims.x_rows, 1);

            for (int i = 0; i < x.rows(); ++i) {
                offset.setZero();
            
                const Scalar dx = internal::findSuitableH(x(i));
                offset(i, 0) = dx;               
                d.col(i) = (_f(x + offset) - _f(x)) / dx;
            }

            // By definition of cppopt gradient vectors are column vectors, whereas the 
            // Jacobian is defined by having the partial derivates in columns.
            if (d.rows() == 1) {
                return d.transpose();
            } else {
                return d;
            }
        }

    private:
        F _f;
        Dims _dims;
    };

    /** Numerical approximation of the second order partial derivatives. 
      *
      * This method works for real valued functions that are either
      * univariate or multivariate.
      *
      * This method repeatedly calls ApproximateForwardDerivative<1> using first
      * order derivative approximation of the input function.
    template<>
    class ApproximateForwardDerivative<2> {
    public:
        
        ApproximateForwardDerivative(const F & f, const Dims &dims) 
            : _f(f), _dims(dims)
        {
        }

        Matrix operator()(const Matrix &x) const
        {
            const ApproximateForwardDerivative<1> df(_f, _dims);

            Matrix d(_dims.x_rows, _dims.x_rows);
            Matrix offset(_dims.x_rows, 1);

            for (int i = 0; i < x.rows(); ++i) {
                offset.setZero();
            
                const Scalar dx = internal::findSuitableH(x(i));
                offset(i, 0) = dx;
                d.col(i) = (df(x + offset) - df(x)) / dx;
            }

            // By definition of cppopt gradient vectors are column vectors, whereas the 
            // Jacobian is defined by having the partial derivates in columns.
            if (d.rows() == 1) {
                return d.transpose();
            } else {
                return d;
            }

            return d;
        }

    private:
        F _f;
        Dims _dims;
    };
          */
}

#endif