#cmake_minimum_required(VERSION 3.5)
#project(ConvNetCpp)

set(SOURCE_FILES_SRC main.cpp MNIST.cpp Model.cpp)

add_executable(FULL_CONNECT ${SOURCE_FILES_SRC})
add_executable(CONV ${SOURCE_FILES_SRC})
add_executable(CONV_POOL ${SOURCE_FILES_SRC})


