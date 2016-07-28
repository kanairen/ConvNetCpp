//
// Created by kanairen on 2016/07/23.
//

#ifndef CONVNETCPP_TESTCONVLAYER_H
#define CONVNETCPP_TESTCONVLAYER_H

#include<gtest/gtest.h>
#include "../src/ConvLayer.h"
#include "../src/activation.h"

class ConvLayer2dTest : public ::testing::Test {
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

    ConvLayer2d_ *layer;

    ConvLayer2dTest() : layer(
            new ConvLayer2d_(N_DATA, INPUT_WIDTH, INPUT_HEIGHT, C_IN, C_OUT,
                             KW, KH, SX, SY, PX, PY, ACTIVATION,
                             GRAD_ACTIVATION, IS_WEIGHT_INIT_ENABLED, 1.f)) { }

    virtual ~ConvLayer2dTest() {
        delete layer;
    }

    virtual void SetUp() {
        std::cout << "ConvLayer2dTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "ConvLayer2dTest::TearDown()" << std::endl;
    }

};

const unsigned int ConvLayer2dTest::N_DATA = 1;
const unsigned int ConvLayer2dTest::INPUT_WIDTH = 3;
const unsigned int ConvLayer2dTest::INPUT_HEIGHT = 3;
const unsigned int ConvLayer2dTest::C_IN = 1;
const unsigned int ConvLayer2dTest::C_OUT = 2;
const unsigned int ConvLayer2dTest::KW = 2;
const unsigned int ConvLayer2dTest::KH = 2;
const unsigned int ConvLayer2dTest::SX = 1;
const unsigned int ConvLayer2dTest::SY = 1;
const unsigned int ConvLayer2dTest::PX = 0;
const unsigned int ConvLayer2dTest::PY = 0;

TEST_F(ConvLayer2dTest, test_init) {

    std::cout << "ConvLayer2dTest::test_init()... " << std::endl;

    const MatrixXf &weights = layer->get_weights();
    const VectorXf &biases = layer->get_biases();
    const MatrixXf &delta = layer->get_delta();
    const MatrixXf &z = layer->get_z();

    const unsigned int n_in = layer->get_n_in();
    const unsigned int n_out = layer->get_n_out();

    std::cout << "weights : " << std::endl;
    std::cout << weights << std::endl;

    std::cout << "biases : " << std::endl;
    std::cout << biases << std::endl;

    std::cout << "delta : " << std::endl;
    std::cout << delta << std::endl;

    std::cout << "z : " << std::endl;
    std::cout << z << std::endl;

    ASSERT_EQ(weights.rows(), n_out);
    ASSERT_EQ(weights.cols(), n_in);

    bool is_nonzero_exists;
    for (int i = 0; i < weights.rows(); ++i) {
        is_nonzero_exists = false;
        for (int j = 0; j < weights.cols(); ++j) {
            // TODO 比較方法を修正すべき
            if (weights(i, j) != 0.f) {
                is_nonzero_exists = true;
            }
        }
        ASSERT_TRUE(is_nonzero_exists);
    }

    ASSERT_EQ(biases.size(), n_out);

    ASSERT_EQ(delta.rows(), n_out);
    ASSERT_EQ(delta.cols(), N_DATA);

    ASSERT_EQ(z.rows(), n_out);
    ASSERT_EQ(z.cols(), N_DATA);

}


TEST_F(ConvLayer2dTest, test_backward) {

    std::cout << "ConvLayer2dTest::test_backward()... " << std::endl;

    unsigned int next_n_out = 4;
    unsigned int n_out = layer->get_n_out();
    unsigned int n_in = layer->get_n_in();

    MatrixXf input(INPUT_WIDTH *INPUT_HEIGHT
    *C_IN, 1);
    input << 1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9;

    std::cout << "weights : " << std::endl;
    std::cout << layer->get_weights() << std::endl;

    // test forward
    const MatrixXf &output = layer->forward(input);

    std::cout << "forward : " << std::endl;
    std::cout << output << std::endl;

    MatrixXf result_output(n_out, N_DATA);
    result_output <<
    12,
            16,
            24,
            28,

            12,
            16,
            24,
            28;

    ASSERT_EQ(output, result_output);

    // test backward
    MatrixXf &&next_weight = (MatrixXf &&) MatrixXf::Ones(next_n_out, n_out);
    MatrixXf &&next_delta = (MatrixXf &&) MatrixXf::Ones(next_n_out, N_DATA);

    std::cout << "backward : " << std::endl;
    layer->backward(next_weight, next_delta, input, n_out, 1);

    // test delta
    const MatrixXf &delta = layer->get_delta();

    std::cout << "delta : " << std::endl;
    std::cout << delta << std::endl;

    MatrixXf &&result_delta = (MatrixXf &&) MatrixXf::Constant(n_out, N_DATA,
                                                               4);

    ASSERT_EQ(delta, result_delta);

    //test learned weights
    const MatrixXf &learned_weights = layer->get_weights();

    std::cout << "weights : " << std::endl;
    std::cout << learned_weights << std::endl;

    MatrixXf &&result_learned_weights = (MatrixXf &&) MatrixXf(n_out, n_in);
    result_learned_weights << -47, -63, 0, -95, -111, 0, 0, 0, 0,
            0, -47, -63, 0, -95, -111, 0, 0, 0,
            0, 0, 0, -47, -63, 0, -95, -111, 0,
            0, 0, 0, 0, -47, -63, 0, -95, -111,
            -47, -63, 0, -95, -111, 0, 0, 0, 0,
            0, -47, -63, 0, -95, -111, 0, 0, 0,
            0, 0, 0, -47, -63, 0, -95, -111, 0,
            0, 0, 0, 0, -47, -63, 0, -95, -111;

    ASSERT_EQ(learned_weights, result_learned_weights);

    // test learned biases
    const VectorXf &learned_biases = layer->get_biases();

    std::cout << "biases : " << std::endl;
    std::cout << learned_biases << std::endl;

    VectorXf &&result_biases = (VectorXf &&) VectorXf(n_out);
    result_biases << -4, -4, -4, -4, -4, -4, -4, -4;

    ASSERT_EQ(learned_biases, result_biases);

    // check forward after learning
    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

}


#endif //CONVNETCPP_TESTCONVLAYER_H
