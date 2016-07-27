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

    virtual void SetUp() {
        std::cout << "TestLayer::SetUp()" << std::endl;
    }

public:

    unsigned int n_data;
    unsigned int n_in;
    unsigned int n_out;

    float (*activation)(float);

    float (*grad_activation)(float);

    LayerTest() : n_data(5), n_in(3), n_out(4),
                  activation(iden), grad_activation(g_iden),
                  layer(new Layer_(5, 3, 4, iden, g_iden, false, 1.f)) { }

    ~LayerTest() {
        delete layer;
    }

};

TEST_F(LayerTest, test_init) {

    std::cout << "TestLayer::test_init()... " << std::endl;

    const MatrixXf &weights = layer->get_weights();
    const VectorXf &biases = layer->get_biases();
    const MatrixXf &delta = layer->get_delta();
    const MatrixXf &z = layer->get_z();

    ASSERT_EQ(weights.rows(), n_out);
    ASSERT_EQ(weights.cols(), n_in);

    ASSERT_EQ(biases.size(), n_out);

    ASSERT_EQ(delta.rows(), n_out);
    ASSERT_EQ(delta.cols(), n_data);

    ASSERT_EQ(z.rows(), n_out);
    ASSERT_EQ(z.cols(), n_data);

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

    MatrixXf input(n_in, n_data);
    input << 1, 4, 7, 10, 13,
            2, 5, 8, 11, 14,
            3, 6, 9, 12, 15;

    const MatrixXf &output = layer->forward(input);

    std::cout << "output : " << std::endl;
    std::cout << output << std::endl;

    MatrixXf result_forward(n_out, n_data);
    result_forward << 3, 7, 11,
            3, 7, 11,
            3, 7, 11,
            3, 7, 11;

    ASSERT_EQ(output, result_forward);

}

TEST_F(LayerTest, test_backward) {

    std::cout << "TestLayer::test_backward()... " << std::endl;

    MatrixXf input(n_in, n_data);
    input << 1, 3, 5,
            2, 4, 6;

    const unsigned int next_n_out = 4;

    MatrixXf &&next_weight = (MatrixXf &&) MatrixXf::Ones(next_n_out, n_out);
    MatrixXf &&next_delta = (MatrixXf &&) MatrixXf::Ones(next_n_out, n_data);

    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

    std::cout << "backward : " << std::endl;
    layer->backward(next_weight, next_delta, input, n_out, 1.f);

    // test for backward
    MatrixXf &&result_delta = (MatrixXf &&) MatrixXf::Constant(n_out, n_data,
                                                               6.f);

    ASSERT_EQ(layer->get_delta(), result_delta);

    MatrixXf &&result_weight = (MatrixXf &&) MatrixXf(n_out, n_in);
    result_weight << -17, -23,
            -17, -23,
            -17, -23,
            -17, -23;

    ASSERT_EQ(layer->get_weights(), result_weight);

    VectorXf &&result_biases = (VectorXf &&) VectorXf(n_out);
    result_biases << -6, -6, -6, -6;

    ASSERT_EQ(layer->get_biases(), result_biases);

    // test for forward again.

    MatrixXf result_forward(n_out, n_data);
    result_forward << -69, -149, -229,
            -69, -149, -229,
            -69, -149, -229,
            -69, -149, -229;

    ASSERT_EQ(layer->forward(input), result_forward);

    std::cout << "forward : " << std::endl;
    std::cout << layer->forward(input) << std::endl;

}

#endif //CONVNETCPP_TESTLAYER_H

