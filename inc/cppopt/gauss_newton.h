// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.


#ifndef CPPOPT_GAUSS_NEWTON
#define CPPOPT_GAUSS_NEWTON

#include "types.h"

namespace cppopt {
    
    
    void gaussNewton(const F &f, const F &d, Matrix &x) {
        Matrix j = d(x);
        assert(j.rows() >= j.cols());
        
        Matrix jt = j.transpose();
        Matrix y = f(x);
        Matrix s = (jt * j).llt().solve(jt * y * Scalar(-1));
        
        x = x + s;
    }
}

#endif