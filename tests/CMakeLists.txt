#cmake_minimum_required(VERSION 3.5)
#project(ConvNetCpp)

set(SOURCE_FILES_TESTS suite.cpp TestMNIST.cpp TestModel.cpp)

add_executable(tests ${SOURCE_FILES_TESTS})

