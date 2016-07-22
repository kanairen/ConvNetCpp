//
// Created by 金井廉 on 2016/07/21.
//

#ifndef CONVNETCPP_TESTLAYER_H
#define CONVNETCPP_TESTLAYER_H

#include "../src/activation.h"
#include "../src/Layer.h"

#undef NODEBUG


void test_init() {

    std::cout << "TestLayer::test_init()... " << std::endl;

    const unsigned int n_data = 5;
    const unsigned int n_in = 3;
    const unsigned int n_out = 4;
    float (*activation)(float) = iden;
    float (*grad_activation)(float) = g_iden;

    Layer_ layer(n_data, n_in, n_out, activation, grad_activation);

    const MatrixXf &weights = layer.get_weights();
    const VectorXf &biases = layer.get_biases();
    const MatrixXf &delta = layer.get_delta();
    const MatrixXf &z = layer.get_z();

    assert(weights.rows() == n_out);
    assert(weights.cols() == n_in);

    assert(biases.size() == n_out);

    assert(delta.rows() == n_out);
    assert(delta.cols() == n_data);

    assert(z.rows() == n_out);
    assert(z.cols() == n_data);

    std::cout << "weights : " << std::endl;
    std::cout << weights << std::endl;

    std::cout << "biases : " << std::endl;
    std::cout << layer.get_biases() << std::endl;

    std::cout << "delta : " << std::endl;
    std::cout << layer.get_delta() << std::endl;

    std::cout << "z : " << std::endl;
    std::cout << layer.get_z() << std::endl;

}

void test_forward() {

    std::cout << "TestLayer::test_forward()... " << std::endl;

    const unsigned int n_data = 5;
    const unsigned int n_in = 3;
    const unsigned int n_out = 4;
    float (*activation)(float) = iden;
    float (*grad_activation)(float) = g_iden;

    Layer_ layer(n_data, n_in, n_out, activation, grad_activation, false, 1);

    MatrixXf input(n_in, n_data);
    input << 1, 4, 7, 10, 13,
            2, 5, 8, 11, 14,
            3, 6, 9, 12, 15;

    std::cout << "output : " << std::endl;
    std::cout << layer.forward(input) << std::endl;


}

void test_backward() {

    std::cout << "TestLayer::test_backward()... " << std::endl;

    const unsigned int n_data = 3;
    const unsigned int n_in = 2;
    const unsigned int n_out = 4;
    const unsigned int next_n_out = 6;
    float (*activation)(float) = iden;
    float (*grad_activation)(float) = g_iden;

    Layer_ layer(n_data, n_in, n_out, activation, grad_activation, false, 1);

    MatrixXf input(n_in, n_data);
    input << 1, 3, 5,
            2, 4, 6;

    MatrixXf &&next_weight = (MatrixXf &&) MatrixXf::Ones(next_n_out, n_out);
    MatrixXf &&next_delta = (MatrixXf &&) MatrixXf::Ones(next_n_out, n_data);

    std::cout << "forward : " << std::endl;
    std::cout << layer.forward(input) << std::endl;

    // test for forward
    MatrixXf result_forward(n_out, n_data);
    result_forward << 3, 7, 11,
            3, 7, 11,
            3, 7, 11,
            3, 7, 11;

    assert(layer.forward(input) == result_forward);

    std::cout << "backward : " << std::endl;
    layer.backward(next_weight, next_delta, input, n_out, 1);

    // test for backward
    MatrixXf &&result_delta = (MatrixXf &&) MatrixXf::Constant(n_out, n_data,
                                                               6);

    assert(layer.get_delta() == result_delta);

    MatrixXf &&result_weight = (MatrixXf &&) MatrixXf(n_out, n_in);
    result_weight << -17, -23,
            -17, -23,
            -17, -23,
            -17, -23;

    assert(layer.get_weights() == result_weight);

    VectorXf &&result_biases = (VectorXf &&) VectorXf(n_out);
    result_biases << -6, -6, -6, -6;

    assert(layer.get_biases() == result_biases);


    // test for forward again.

    result_forward << -69, -149, -229,
            -69, -149, -229,
            -69, -149, -229,
            -69, -149, -229;

    assert(layer.forward(input) == result_forward);

    std::cout << "forward : " << std::endl;
    std::cout << layer.forward(input) << std::endl;


}

#endif //CONVNETCPP_TESTLAYER_H
