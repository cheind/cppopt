# cppopt

cppopt is lightweight library for numerical optimization in C++11. The main focus of this library is to provide a concise and easy to follow implementations of algorithms.

## Coverage

Currently the following algorithms are implemented for univariate and multivariate functions
 - Gradient Descent
 - Newton-Raphson
 - Gauss-Newton for non-linear least squares.

## Usage

With cppopt functions and derivatives are defined through lambda expression. For example the lambda expression for evaluating the gradient of the multivariate polynom
    
    f(x,y) = x^2 + y^2 + 2x + 8y

with 

    df/dx = 2x + 2
    df/dy = 2y + 8

is given by
```
cppopt::F df = [](const cppopt::Matrix &x) -> cppopt::Matrix {
    cppopt::Matrix d(2, 1);
        
    d(0) = 2.f * x(0) + 2.f;
    d(1) = 2.f * x(1) + 8.f;
        
    return d;
};
```

## Links
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