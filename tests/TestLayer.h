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
        std::cout << "LayerTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "LayerTest::TearDown()" << std::endl;
    }

};

// static constant values
const unsigned int LayerTest::N_DATA = 5;
const unsigned int LayerTest::N_IN = 3;
const unsigned int LayerTest::N_OUT = 4;
const unsigned int LayerTest::NEXT_N_OUT = 4;

TEST_F(LayerTest, test_init) {

    std::cout << "LayerTest::test_init()... " << std::endl;

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

    std::cout << "LayerTest::test_forward()... " << std::endl;

    MatrixXf input(N_IN, N_DATA);
    input << 1, 4, 7, 10, 13,
            2, 5, 8, 11, 14,
            3, 6, 9, 12, 15;

    const MatrixXf &output = layer->forward(input);

    std::cout << "output : " << std::endl;
    std::cout << output << std::endl;

    MatrixXf result_forward(N_OUT, N_DATA);
    result_forward << 6, 15, 24, 33, 42,
            6, 15, 24, 33, 42,
            6, 15, 24, 33, 42,
            6, 15, 24, 33, 42;

    ASSERT_EQ(output, result_forward);

}

TEST_F(LayerTest, test_backward) {

    std::cout << "LayerTest::test_backward()... " << std::endl;

    MatrixXf input(N_IN, N_DATA);
    input << 1, 4, 7, 10, 13,
            2, 5, 8, 11, 14,
            3, 6, 9, 12, 15;

    MatrixXf &&next_weight = (MatrixXf &&) MatrixXf::Ones(NEXT_N_OUT, N_OUT);
    MatrixXf &&next_delta = (MatrixXf &&) MatrixXf::Ones(NEXT_N_OUT, N_DATA);

    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

    std::cout << "backward..." << std::endl;
    layer->backward(next_weight, next_delta, input, NEXT_N_OUT, 1.f);

    // test for backward delta
    const MatrixXf &delta = layer->get_delta();

    MatrixXf &&result_delta = (MatrixXf &&) MatrixXf::Constant(N_OUT, N_DATA,
                                                               4.f);

    std::cout << "delta : " << std::endl;
    std::cout << delta << std::endl;

    ASSERT_EQ(delta, result_delta);


    // test for learned weights
    const MatrixXf &learned_weights = layer->get_weights();

    std::cout << "learned_weights : " << std::endl;
    std::cout << learned_weights << std::endl;

    MatrixXf &&result_weight = (MatrixXf &&) MatrixXf(N_OUT, N_IN);
    result_weight << -27, -31, -35,
            -27, -31, -35,
            -27, -31, -35,
            -27, -31, -35;

    ASSERT_EQ(layer->get_weights(), result_weight);

    // test for learned biases
    const VectorXf &learned_biases = layer->get_biases();

    std::cout << "learned_biases : " << std::endl;
    std::cout << learned_biases << std::endl;

    VectorXf &&result_biases = (VectorXf &&) VectorXf(N_OUT);
    result_biases << -4, -4, -4, -4;

    ASSERT_EQ(learned_biases, result_biases);

    // test for forward again.
    const MatrixXf &output = layer->forward(input);

    std::cout << "forward : " << std::endl;
    std::cout << output << std::endl;

    MatrixXf result_forward(N_OUT, N_DATA);
    result_forward << -198, -477, -756, -1035, -1314,
            -198, -477, -756, -1035, -1314,
            -198, -477, -756, -1035, -1314,
            -198, -477, -756, -1035, -1314;

    ASSERT_EQ(output, result_forward);

}

#endif //CONVNETCPP_TESTLAYER_H

