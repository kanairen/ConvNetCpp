#cmake_minimum_required(VERSION 3.5)
#project(ConvNetCpp)

set(SOURCE_FILES_SRC main.cpp MNIST.cpp tinyxml/tinyxml2.cpp)

add_executable(FULL_CONNECT ${SOURCE_FILES_SRC})
add_executable(CONV ${SOURCE_FILES_SRC})
add_executable(CONV_POOL ${SOURCE_FILES_SRC})

add_executable(SHAPE_MAP_CNN ${SOURCE_FILES_SRC})


