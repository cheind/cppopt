// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.

#ifndef CPPOPT_NEWTON_RAPHSON
#define CPPOPT_NEWTON_RAPHSON

#include "types.h"

namespace cppopt {
    
    void newtonRaphson(const F &f, const F &d, Matrix &x) {
        Matrix jacobian = d(x);
        assert(jacobian.rows() == jacobian.cols());
        Matrix y = f(x);
        Matrix s = jacobian.lu().solve(y * Scalar(-1));
        x = x + s;
    }
}

#endif