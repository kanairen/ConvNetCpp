cmake_minimum_required(VERSION 3.5)
project(ConvNetCpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    ConvNetCpp/Activation.h
    ConvNetCpp/Layer.cpp
    ConvNetCpp/Layer.h
    ConvNetCpp/main.cpp
    ConvNetCpp/MNIST.cpp
    ConvNetCpp/MNIST.h
    ConvNetCpp/Model.cpp
    ConvNetCpp/Model.h
    ConvNetCpp/test.cpp
    ConvNetCpp/test.h)

include_directories("/usr/local/include/eigen3")

add_executable(ConvNetCpp ${SOURCE_FILES})