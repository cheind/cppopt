// This file is part of cppopt, a lightweight C++ library
// for numerical optimization
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD 3 license.
// If a copy of the MPL was not distributed with this file, You can obtain
// one at http://opensource.org/licenses/BSD-3-Clause.


#ifndef CPPOPT_GRADIENT_DESCENT
#define CPPOPT_GRADIENT_DESCENT

#include "types.h"

namespace cppopt {
    
    void gradientDescent(const F &d, Matrix &x, Scalar step) {
        x = x - step * d(x);
    }
}

#endif