//
// Created by kanairen on 2016/07/21.
//

#ifndef CONVNETCPP_TESTLAYER_H
#define CONVNETCPP_TESTLAYER_H

#include<gtest/gtest.h>
#include "../src/activation.h"
#include "../src/Layer.h"


class LayerTest : public ::testing::Test {
protected:

    Layer_ *layer;

    static const unsigned int N_DATA;
    static const unsigned int N_IN;
    static const unsigned int N_OUT;
    static const unsigned int NEXT_N_OUT;

    constexpr static float (*const ACTIVATION)(float) = iden;

    constexpr static float (*const GRAD_ACTIVATION)(float) = g_iden;

    LayerTest() : layer(new Layer_(N_DATA, N_IN, N_OUT,
                                   ACTIVATION, GRAD_ACTIVATION, false, 1.f)) { }

    virtual ~LayerTest() {
        delete layer;
    }

    virtual void SetUp() {
        std::cout << "TestLayer::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "TestLayer::TearDown()" << std::endl;
    }

};

// static constant values
const unsigned int LayerTest::N_DATA = 5;
const unsigned int LayerTest::N_IN = 3;
const unsigned int LayerTest::N_OUT = 4;
const unsigned int LayerTest::NEXT_N_OUT = 4;

TEST_F(LayerTest, test_init) {

    std::cout << "TestLayer::test_init()... " << std::endl;

    const MatrixXf &weights = layer->get_weights();
    const VectorXf &biases = layer->get_biases();
    const MatrixXf &delta = layer->get_delta();
    const MatrixXf &z = layer->get_z();

    ASSERT_EQ(weights.rows(), N_OUT);
    ASSERT_EQ(weights.cols(), N_IN);

    ASSERT_EQ(biases.size(), N_OUT);

    ASSERT_EQ(delta.rows(), N_OUT);
    ASSERT_EQ(delta.cols(), N_DATA);

    ASSERT_EQ(z.rows(), N_OUT);
    ASSERT_EQ(z.cols(), N_DATA);

    std::cout << "weights : " << std::endl;
    std::cout << weights << std::endl;

    std::cout << "biases : " << std::endl;
    std::cout << biases << std::endl;

    std::cout << "delta : " << std::endl;
    std::cout << delta << std::endl;

    std::cout << "z : " << std::endl;
    std::cout << z << std::endl;

}

TEST_F(LayerTest, test_forward) {

    std::cout << "TestLayer::test_forward()... " << std::endl;

    MatrixXf input(N_IN, N_DATA);
    input << 1, 4, 7, 10, 13,
            2, 5, 8, 11, 14,
            3, 6, 9, 12, 15;

    const MatrixXf &output = layer->forward(input);

    std::cout << "output : " << std::endl;
    std::cout << output << std::endl;

    MatrixXf result_forward(N_OUT, N_DATA);
    result_forward << 3, 7, 11,
            3, 7, 11,
            3, 7, 11,
            3, 7, 11;

    ASSERT_EQ(output, result_forward);

}

TEST_F(LayerTest, test_backward) {

    std::cout << "TestLayer::test_backward()... " << std::endl;

    MatrixXf input(N_IN, N_DATA);
    input << 1, 3, 5,
            2, 4, 6;

    MatrixXf &&next_weight = (MatrixXf &&) MatrixXf::Ones(NEXT_N_OUT, N_OUT);
    MatrixXf &&next_delta = (MatrixXf &&) MatrixXf::Ones(NEXT_N_OUT, N_DATA);

    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

    std::cout << "backward : " << std::endl;
    layer->backward(next_weight, next_delta, input, N_OUT, 1.f);

    // test for backward
    MatrixXf &&result_delta = (MatrixXf &&) MatrixXf::Constant(N_OUT, N_DATA,
                                                               6.f);

    ASSERT_EQ(layer->get_delta(), result_delta);

    MatrixXf &&result_weight = (MatrixXf &&) MatrixXf(N_OUT, N_IN);
    result_weight << -17, -23,
            -17, -23,
            -17, -23,
            -17, -23;

    ASSERT_EQ(layer->get_weights(), result_weight);

    VectorXf &&result_biases = (VectorXf &&) VectorXf(N_OUT);
    result_biases << -6, -6, -6, -6;

    ASSERT_EQ(layer->get_biases(), result_biases);

    // test for forward again.

    MatrixXf result_forward(N_OUT, N_DATA);
    result_forward << -69, -149, -229,
            -69, -149, -229,
            -69, -149, -229,
            -69, -149, -229;

    ASSERT_EQ(layer->forward(input), result_forward);

    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

}

#endif //CONVNETCPP_TESTLAYER_H

