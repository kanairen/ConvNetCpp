//
// Created by kanairen on 2016/07/23.
//

#ifndef CONVNETCPP_TESTGRIDLAYER_H
#define CONVNETCPP_TESTGRIDLAYER_H

#include<gtest/gtest.h>
#include "../src/GridLayer.h"

class GridLayer2dTest : public ::testing::Test {
protected:
    static const unsigned int N_DATA;
    static const unsigned int INPUT_WIDTH;
    static const unsigned int INPUT_HEIGHT;
    static const unsigned int C_IN;
    static const unsigned int C_OUT;
    static const unsigned int KW;
    static const unsigned int KH;
    static const unsigned int SX;
    static const unsigned int SY;
    static const unsigned int PX;
    static const unsigned int PY;

    constexpr static float (*const ACTIVATION)(float) = iden;

    constexpr static float (*const GRAD_ACTIVATION)(float) = g_iden;

    static const bool IS_WEIGHT_INIT_ENABLED = false;

    GridLayer2d_ *layer;

    GridLayer2dTest() : layer(
            new GridLayer2d_(N_DATA, INPUT_WIDTH, INPUT_HEIGHT, C_IN, C_OUT,
                             KW, KH, SX, SY, PX, PY, ACTIVATION,
                             GRAD_ACTIVATION, IS_WEIGHT_INIT_ENABLED)) { }

    virtual ~GridLayer2dTest() {
        delete layer;
    }

    virtual void SetUp() {
        std::cout << "GridLayer2dTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "GridLayer2dTest::TearDown()" << std::endl;
    }

public:
};

const unsigned int GridLayer2dTest::N_DATA = 3;
const unsigned int GridLayer2dTest::INPUT_WIDTH = 3;
const unsigned int GridLayer2dTest::INPUT_HEIGHT = 3;
const unsigned int GridLayer2dTest::C_IN = 1;
const unsigned int GridLayer2dTest::C_OUT = 2;
const unsigned int GridLayer2dTest::KW = 2;
const unsigned int GridLayer2dTest::KH = 2;
const unsigned int GridLayer2dTest::SX = 1;
const unsigned int GridLayer2dTest::SY = 1;
const unsigned int GridLayer2dTest::PX = 0;
const unsigned int GridLayer2dTest::PY = 0;


TEST_F(GridLayer2dTest, test_filter_outsize) {

    ASSERT_EQ(GridLayer2d_::filter_outsize(28, KW, SX, PX, false), 27);
    ASSERT_EQ(GridLayer2d_::filter_outsize(28, KH, SY, PY, false), 27);
}


#endif //CONVNETCPP_TESTGRIDLAYER_H
