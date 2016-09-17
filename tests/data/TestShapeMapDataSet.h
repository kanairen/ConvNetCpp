//
// Created by Ren Kanai on 2016/09/17.
//

#ifndef CONVNETCPP_TESTSHAPEMAPDATASET_H
#define CONVNETCPP_TESTSHAPEMAPDATASET_H

#include "gtest/gtest.h"
#include "../../src/data/ShapeMapDataSet.h"

class ShapeMapDataSetTest : public ::testing::Test {
protected:

    ShapeMapDataSetTest() { }

    virtual ~ShapeMapDataSetTest() { }

    virtual void SetUp() {
        std::cout << "ShapeMapDataSetTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "ShapeMapDataSet::TearDown()" << std::endl;
    }

};

TEST_F(ShapeMapDataSetTest, test_load) {
    std::cout << "ShapeMapDataSetTest::listdir()" << std::endl;
    ShapeMapDataSetTest::load
}

#endif //CONVNETCPP_TESTSHAPEMAPDATASET_H
