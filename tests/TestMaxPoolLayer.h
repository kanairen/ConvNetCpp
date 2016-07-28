//
// Created by kanairen on 2016/07/25.
//

#ifndef CONVNETCPP_TESTMAXPOOLLAYER_H
#define CONVNETCPP_TESTMAXPOOLLAYER_H

#include<gtest/gtest.h>
#include "../src/MaxPoolLayer.h"

class MaxPoolLayer2dTest : public ::testing::Test {
protected:
    static const unsigned int N_DATA;
    static const unsigned int INPUT_WIDTH;
    static const unsigned int INPUT_HEIGHT;
    static const unsigned int C_IN;
    static const unsigned int KW;
    static const unsigned int KH;
    static const unsigned int PX;
    static const unsigned int PY;

    constexpr static float (*const ACTIVATION)(float) = iden;

    constexpr static float (*const GRAD_ACTIVATION)(float) = g_iden;

    static const bool IS_WEIGHT_INIT_ENABLED = false;

    MaxPoolLayer2d_ *layer;

    MaxPoolLayer2dTest() : layer(
            new MaxPoolLayer2d_(N_DATA, INPUT_WIDTH, INPUT_HEIGHT, C_IN, KW, KH,
                                PX, PY)) { }

    virtual ~MaxPoolLayer2dTest() {
        delete layer;
    }

    virtual void SetUp() {
        std::cout << "MaxPoolLayer2dTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "MaxPoolLayer2dTest::TearDown()" << std::endl;
    }

};

const unsigned int MaxPoolLayer2dTest::N_DATA = 1;
const unsigned int MaxPoolLayer2dTest::INPUT_WIDTH = 4;
const unsigned int MaxPoolLayer2dTest::INPUT_HEIGHT = 4;
const unsigned int MaxPoolLayer2dTest::C_IN = 1;
const unsigned int MaxPoolLayer2dTest::KW = 2;
const unsigned int MaxPoolLayer2dTest::KH = 2;
const unsigned int MaxPoolLayer2dTest::PX = 0;
const unsigned int MaxPoolLayer2dTest::PY = 0;


TEST_F(MaxPoolLayer2dTest, test_init) {

    std::cout << "MaxPoolLayer2dTest::test_init()... " << std::endl;

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

    ASSERT_EQ(biases.size(), n_out);

    ASSERT_EQ(delta.rows(), n_out);
    ASSERT_EQ(delta.cols(), N_DATA);

    ASSERT_EQ(z.rows(), n_out);
    ASSERT_EQ(z.cols(), N_DATA);

}


TEST_F(MaxPoolLayer2dTest,test_forward) {

    std::cout << "MaxPoolLayer2dTest::test_forward()... " << std::endl;

    const unsigned int n_in = layer->get_n_in();

    MatrixXf input(n_in, N_DATA);
    input << 1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16;

    std::cout << "output : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

}

TEST_F(MaxPoolLayer2dTest, test_backward) {

    std::cout << "TMaxPoolLayer2dTest::test_backward()... " << std::endl;

    unsigned int next_n_out = 4;
    unsigned int n_out = layer->get_n_out();
    unsigned int n_in = layer->get_n_in();

    MatrixXf input(n_in, N_DATA);
    input << 1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16;

    std::cout << "weights : " << std::endl;
    std::cout << layer->get_weights() << std::endl;

    MatrixXf &&next_weight = (MatrixXf &&) MatrixXf::Ones(next_n_out,
                                                          n_out);
    MatrixXf &&next_delta = (MatrixXf &&) MatrixXf::Ones(next_n_out,
                                                         N_DATA);

    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

    // test for forward
    MatrixXf result_forward(n_out, N_DATA);
    result_forward << 6,
            14,
            8,
            16;

    ASSERT_EQ(layer->forward(input) , result_forward);

    std::cout << "backward : " << std::endl;
    layer->backward(next_weight, next_delta, input, n_out, 1);

    std::cout << "delta : " << std::endl;
    std::cout << layer->get_delta() << std::endl;

    // test for backward
    MatrixXf result_delta(n_out, N_DATA);
    result_delta << 4,
            4,
            4,
            4;

    assert(layer->get_delta() == result_delta);

    MatrixXf &&res_w = (MatrixXf &&) MatrixXf(n_out, n_in);
    res_w << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

    std::cout << "weights : " << std::endl;
    std::cout << layer->get_weights() << std::endl;

    assert(layer->get_weights() == res_w);

    VectorXf &&result_biases = (VectorXf &&) VectorXf(n_out);
    result_biases << 0, 0, 0, 0;

    std::cout << "biases : " << std::endl;
    std::cout << layer->get_biases() << std::endl;

    assert(layer->get_biases() == result_biases);

    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

}

#endif //CONVNETCPP_TESTMAXPOOLLAYER_H
