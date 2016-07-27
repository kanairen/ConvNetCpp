//
// Created by kanairen on 2016/06/14.
//

#include "TestLayer.h"
#include "TestSoftMaxLayer.h"
#include "TestGridLayer.h"
#include "TestConvLayer.h"
#include "TestMaxPoolLayer.h"

#include "TestModel.h"

int main(int argc, char **argv) {

    // Google Test
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}
