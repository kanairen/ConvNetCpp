cmake_minimum_required(VERSION 3.5)
project(ConvNetCpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("/usr/local/include/eigen3")

add_subdirectory(src)
add_subdirectory(tests)


