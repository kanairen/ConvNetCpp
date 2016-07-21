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

    const unsigned int n_data = 10;
    const unsigned int n_in = 3;
    const unsigned int n_out = 4;
    float (*activation)(float) = iden;
    float (*grad_activation)(float) = g_iden;

    Layer_ layer(n_data, n_in, n_out, activation, g_iden);

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

}

#endif //CONVNETCPP_TESTLAYER_H
