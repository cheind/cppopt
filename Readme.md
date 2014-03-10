# cppopt

cppopt is lightweight library for numerical optimization in C++11. The main focus of this library is to provide concise and easy to follow implementations of various optimization algorithms. The library can easily be integrated into other projects as it is header only.

## Coverage

Currently the following algorithms are implemented for univariate and multivariate functions
 - Gradient Descent
 - Newton-Raphson
 - Gauss-Newton for non-linear least squares.

## Conventions

Let *f* be a [function](http://en.wikipedia.org/wiki/Function_(mathematics)) with N dimensional input variables and M dimensional outputs. 
We say that 
 - *f* is univariate if `N == 1` and multivariate if `N > 1`
 - *f* is scalar valued if `M == 1` and vector valued if `M > 1`

The first order derivatives of a multivariate scalar valued function is called the [gradient vector](http://en.wikipedia.org/wiki/Gradient). 
The first order derivatives of a vector valued function is called the [Jacobian](http://de.wikipedia.org/wiki/Jacobi-Matrix) matrix. 
The second order derivatives of a multivariate scalar valued function is called the [Hessian](http://en.wikipedia.org/wiki/Hessian_matrix) matrix.

Throughout cppopt we use the following conventions
 - input variables are repesented by `Nx1` column vectors
 - outputs are represented by `Mx1` column vectors
 - gradients are represented by `Nx1` column vectors
 - Jacobians are represented by `MxN` matrices
 - Hessians are represented by `NxN` matrices

## Usage

With cppopt functions and derivatives are defined through lambda expression. For example the lambda expression for evaluating the gradient of the multivariate polynom
    
    f(x,y) = x^2 + y^2 + 2x + 8y

with 

    df/dx = 2x + 2
    df/dy = 2y + 8

is given by

    // Gradient of f(x,y) = x^2 + y^2 + 2x + 8y
    cppopt::F df = [](const cppopt::Matrix &x) -> cppopt::Matrix {
        cppopt::Matrix d(2, 1);
        
        d(0) = 2.f * x(0) + 2.f;
        d(1) = 2.f * x(1) + 8.f;
        
        return d;
    };


This lambda expression can then be passed as an argument to various optimization routines as shown below.

    // Start solution
    cppopt::Matrix x(2, 1);
    x(0) = -3.f;
    x(1) = -2.f;

    // Perform one step using gradient descent
    cppopt::gradientDescent(df, x, 0.01f);


## Dependencies

The only dependency for cppopt is the header only Eigen library (http://eigen.tuxfamily.org).

## Unclassified Links
Links of interest I found during the course of developing cppopt

http://pages.cs.wisc.edu/~ferris/cs730/chap3.pdf
http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
http://www.itl.nist.gov/div898/handbook/pmd/section1/pmd143.htm
http://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8633&rep=rep1&type=pdf
http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf

Regularization and weighting
http://see.stanford.edu/materials/lsoeldsee263/07-ls-reg.pdf

Convergence issues:
https://support.sas.com/documentation/cdl/en/etsug/60372/HTML/default/viewer.htm#etsug_model_sect039.htm
http://www.trentfguidry.net/post/2009/08/12/Nonlinear-Least-Squares-Regression-Levenberg-Marquardt.aspx