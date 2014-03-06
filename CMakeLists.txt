cmake_minimum_required(VERSION 2.8)

project(cppopt)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")

set(CPPOPT_HEADERS
    inc/cppopt/types.h
    inc/cppopt/newton_raphson.h
    inc/cppopt/gradient_descent.h
    inc/cppopt/gauss_newton.h
)

# Adapt this string to where your Eigen include directory is located
include_directories(../eigen)
include_directories(inc)

add_executable(2d_gradientdescent ${CPPOPT_HEADERS} test/2d_gradientdescent.cpp)
add_executable(1d_newtonraphson ${CPPOPT_HEADERS} test/1d_newtonraphson.cpp)
add_executable(2d_newtonraphson ${CPPOPT_HEADERS} test/2d_newtonraphson.cpp)
add_executable(nlls_gaussnewton_fitcircle ${CPPOPT_HEADERS} test/nlls_gaussnewton_fitcircle.cpp)
add_executable(nlls_gaussnewton_findrotation ${CPPOPT_HEADERS} test/nlls_gaussnewton_findrotation.cpp)