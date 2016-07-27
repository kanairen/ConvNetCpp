cmake_minimum_required(VERSION 3.5)
#project(ConvNetCpp)

set(SOURCE_FILES_TESTS suite.cpp TestConvLayer.h TestMaxPoolLayer.h)
# TODO パスを一般化
set(GOOGLE_TEST_ROOT $ENV{HOME}/Projects/cpp/ConvNetCpp/googletest)

include_directories(${GOOGLE_TEST_ROOT}/include)
add_executable(tests ${SOURCE_FILES_TESTS})
target_link_libraries(
        tests
        pthread
        ${GOOGLE_TEST_ROOT}/build/libgtest.a
        ${GOOGLE_TEST_ROOT}/build/libgtest_main.a
)

