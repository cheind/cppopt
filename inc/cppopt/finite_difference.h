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
            
            const T eps = static_cast<T>(sqrt(std::numeric_limits<T>::epsilon()));
            const T h = eps * x;
            volatile T xph = x + h;
            return xph - x;   
        }

        /** Utility class to evaluate functions at differences specified by offsets */
        class FiniteDifferenceHelper {
        public:
            FiniteDifferenceHelper(const F &f, const Dims &dims)
                : _f(f), _offset(dims.x_rows, 1)
            {}

            Matrix operator()(const Matrix &x, int dim, Scalar h) {
                _offset.setZero();
                _offset(dim) = h;
                return _f(x + _offset);
            }

            Matrix operator()(const Matrix &x, int dim0, Scalar h0, int dim1, Scalar h1) {
                _offset.setZero();
                _offset(dim0) = h0;
                _offset(dim1) = h1;
                return _f(x + _offset);
            }

            Matrix operator()(const Matrix &x) {
                return _f(x);
            }

        private:
            const F &_f;
            Matrix _offset;
        };
    }
    
    /** Approximates the nth order derivative for a multivariate scalar valued function using forward difference. */
    template<int Order>
    class ApproximateCentralDerivative {};

    /** Numerical approximation of the first order partial derivatives. 
      *
      * This method works for real and vector valued functions that are either
      * univariate or multivariate.
      */
    template<>
    class ApproximateCentralDerivative<1> {
    public:
        
        /** Initialize */
        ApproximateCentralDerivative(const F & f, const Dims &dims) 
            : _f(f), _dims(dims)
        {}

        /** Calculate the first order derivative around x.*/
        Matrix operator()(const Matrix &x) const
        {
            Matrix d(_dims.y_rows, _dims.x_rows);
            internal::FiniteDifferenceHelper fdh(_f, _dims);

            for (int i = 0; i < x.rows(); ++i) {
                const Scalar dx = internal::findSuitableH(x(i));             
                d.col(i) = (fdh(x, i, dx) - fdh(x, i, -dx)) / (Scalar(2) * dx);
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
      * This method works for real and vector valued functions that are either
      * univariate or multivariate.
    template<>
    class ApproximateCentralDerivative<2> {
    public:
        
        ApproximateCentralDerivative(const F & f, const Dims &dims) 
            : _f(f), _dims(dims), _offset(dims.x_rows, 1)
        {}

        Matrix operator()(const Matrix &x) const
        {
            Matrix d(_dims.x_rows, _dims.x_rows);
            internal::FiniteDifferenceHelper fdh(_f, _dims);

            for (int r = 0; r < x.rows(); ++r) {
                for (int c = 0; c < x.rows(); ++c) {
                    const Scalar dr = internal::findSuitableH(x(r));                    
                    if (c == r) {
                        d(r, c) = (fdh(x, r, dr) - 
                                   Scalar(2) * fdh(x) + 
                                   fdh(x, r, -dr))(0) / (dr * dr);
                    } else {
                        const Scalar dc = internal::findSuitableH(x(c));
                        d(r, c) = (fdh(x, r, dr, c, dc) - 
                                   fdh(x, r, dr, c, -dc) -
                                   fdh(x, r, -dr, c, dc) +
                                   fdh(x, r, -dr, c, -dc))(0) / (Scalar(4) * dr * dc);
                    }
                }
            }

            return d;
        }

    private:
        F _f;
        Dims _dims;
        Matrix _offset;
    };
    */
}

#endif