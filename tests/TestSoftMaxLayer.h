//
// Created by 金井廉 on 2016/07/22.
//

#ifndef CONVNETCPP_TESTSOFTMAXLAYER_H
#define CONVNETCPP_TESTSOFTMAXLAYER_H

#include "../src/SoftMaxLayer.h"

namespace sm_layer {

    void test_init() {

        const unsigned int n_data = 3;
        const unsigned int n_in = 2;
        const unsigned int n_out = 4;

        SoftMaxLayer_ layer(n_data, n_in, n_out);

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

        std::cout << "TestSoftMaxLayer::test_forward()... " << std::endl;

        const unsigned int n_data = 3;
        const unsigned int n_in = 2;
        const unsigned int n_out = 4;

        SoftMaxLayer_ layer(n_data, n_in, n_out, false, 1.f);

        MatrixXf input(n_in, n_data);
        input << 1, 3, 5,
                2, 4, 6;

        const MatrixXf &output = layer.forward(input);

        MatrixXf result_u(n_out, n_data);
        result_u << 3, 7, 11,
                3, 7, 11,
                3, 7, 11,
                3, 7, 11;

        assert(layer.get_u() == result_u);

        MatrixXf result_z(n_out, n_data);
        result_z << 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25,
                0.25, 0.25, 0.25,
                0.25, 0.25, 0.25;

        assert(output == result_z);

        std::cout << "output : " << std::endl;
        std::cout << output << std::endl;


    }

}

#endif //CONVNETCPP_TESTSOFTMAXLAYER_H
