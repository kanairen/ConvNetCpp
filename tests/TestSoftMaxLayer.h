//
// Created by kanairen on 2016/07/22.
//

#ifndef CONVNETCPP_TESTSOFTMAXLAYER_H
#define CONVNETCPP_TESTSOFTMAXLAYER_H

#include "../src/SoftMaxLayer.h"

class SoftMaxLayerTest : public ::testing::Test {
protected:

    SoftMaxLayer_ *layer;

    static const unsigned int N_DATA;
    static const unsigned int N_IN;
    static const unsigned int N_OUT;

    SoftMaxLayerTest() : layer(
            new SoftMaxLayer_(N_DATA, N_IN, N_OUT, false, 1.f)) { }

    virtual ~SoftMaxLayerTest() { }

    virtual void SetUp() {
        std::cout << "SoftMaxLayerTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "SoftMaxLayerTest::TearDown()" << std::endl;
    }

};

const unsigned int SoftMaxLayerTest::N_DATA = 3;
const unsigned int SoftMaxLayerTest::N_IN = 2;
const unsigned int SoftMaxLayerTest::N_OUT = 4;

TEST_F(SoftMaxLayerTest, test_init) {

    std::cout << "SoftMaxLayerTest::test_init()... " << std::endl;

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

TEST_F(SoftMaxLayerTest, test_forward) {

    std::cout << "SoftMaxLayerTest::test_forward()... " << std::endl;

    // input matrix
    MatrixXf input(N_IN, N_DATA);
    input << 1, 3, 5,
            2, 4, 6;

    // test weighted sum (u) and output (z)
    // do not order of 'z' and 'u'.
    const MatrixXf &z = layer->forward(input);
    const MatrixXf &u = layer->get_u();

    MatrixXf result_u(N_OUT, N_DATA);
    result_u << 3, 7, 11,
            3, 7, 11,
            3, 7, 11,
            3, 7, 11;

    ASSERT_EQ(u, result_u) << "u : " << u << std::endl;

    MatrixXf result_z(N_OUT, N_DATA);
    result_z << 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25,
            0.25, 0.25, 0.25,
            0.25, 0.25, 0.25;

    ASSERT_EQ(z, result_z) << "z : " << z << std::endl;

}

TEST_F(SoftMaxLayerTest, test_backward) {

    std::cout << "SoftMaxLayerTest::test_backward()... " << std::endl;

    MatrixXf dummy_weight(N_OUT, N_IN);
    MatrixXf dummy_prev_output(N_IN, N_DATA);
    MatrixXf &&last_delta = (MatrixXf &&) MatrixXf::Ones(N_OUT, N_DATA);

    const unsigned int next_n_out = 4;

    layer->backward(dummy_weight, last_delta, dummy_prev_output, next_n_out,
                    1.f);

    std::cout << "last_delta : " << std::endl;
    std::cout << last_delta << std::endl;

    std::cout << "delta : " << std::endl;
    std::cout << layer->get_delta() << std::endl;

    // test independence of each delta.
    std::cout << "last_delta(0,0) = 100 " << std::endl;
    last_delta(0, 0) = 100;
    std::cout << last_delta(0, 0) << std::endl;

    std::cout << "delta(0,0) = ..." << std::endl;
    std::cout << layer->get_delta()(0, 0) << std::endl;

    ASSERT_NE(last_delta(0, 0), layer->get_delta()(0, 0));

}

#endif //CONVNETCPP_TESTSOFTMAXLAYER_H
