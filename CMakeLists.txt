cmake_minimum_required(VERSION 3.5)
project(ConvNetCpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        src/main.cpp
        src/Layer.cpp
        src/Layer.h
        src/Model.cpp
        src/Model.h
        src/MNIST.cpp
        src/MNIST.h
        src/activation.h
        src/config.h
        src/Data.h
        src/optimizer.h
        tests/suite.cpp
        tests/TestLayer.cpp
        tests/TestLayer.h
        tests/TestModel.cpp
        tests/TestModel.h
        tests/TestMNIST.cpp
        tests/TestMNIST.h)

include_directories("/usr/local/include/eigen3")

add_executable(ConvNetCpp ${SOURCE_FILES})