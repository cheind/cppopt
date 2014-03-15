cmake_minimum_required(VERSION 2.8)

project(cppopt)

set(EIGEN_INCLUDE_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")

set(CPPOPT_HEADERS
    inc/cppopt/types.h
    inc/cppopt/newton_raphson.h
    inc/cppopt/gradient_descent.h
    inc/cppopt/gauss_newton.h
    inc/cppopt/numerical_derivative.h
)

# Adapt this string to where your Eigen include directory is located
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(inc)

add_executable(2d_gradientdescent ${CPPOPT_HEADERS} examples/2d_gradientdescent.cpp)
add_executable(1d_newtonraphson ${CPPOPT_HEADERS} examples/1d_newtonraphson.cpp)
add_executable(2d_newtonraphson ${CPPOPT_HEADERS} examples/2d_newtonraphson.cpp)
add_executable(nlls_gaussnewton_fitcircle ${CPPOPT_HEADERS} examples/nlls_gaussnewton_fitcircle.cpp)
add_executable(nlls_gaussnewton_findrotation ${CPPOPT_HEADERS} examples/nlls_gaussnewton_findrotation.cpp)

set(CPPOPT_TESTS
    tests/main.cpp
    tests/univariate_function.h
    tests/multivariate_function.h
    tests/test_numerical_derivatives.cpp
    tests/test_gradient_descent.cpp
    tests/test_newton_raphson.cpp
)

include_directories(tests)
add_executable(cppopt_tests ${CPPOPT_HEADERS} ${CPPOPT_TESTS})