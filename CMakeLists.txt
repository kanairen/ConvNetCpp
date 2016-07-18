cmake_minimum_required(VERSION 3.5)
project(ConvNetCpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        src/main.cpp
        src/Model.cpp
        src/MNIST.cpp
        tests/suite.cpp
        tests/TestModel.cpp
        tests/TestMNIST.cpp)

include_directories("/usr/local/include/eigen3")

add_executable(ConvNetCpp ${SOURCE_FILES})