//
// Created by kanairen on 2016/06/14.
//

#include <gtest/gtest.h>
#include "ShareArgs.h"
#include "util/TestIOUtil.h"
#include "util/TestOSUtil.h"
#include "data/TestShapeMapDataSet.h"
//#include "TestLayer.h"
//#include "TestSoftMaxLayer.h"
//#include "TestGridLayer.h"
//#include "TestConvLayer.h"
//#include "TestMaxPoolLayer.h"
//#include "TestModel.h"


int main(int argc, char **argv) {

    // Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Share arguments.
    ARGC = argc;
    ARGV = argv;

    PATH_ROOT = std::string(argv[1]);
    PATH_SHAPE_MAP = std::string(argv[2]);

    return RUN_ALL_TESTS();

}
