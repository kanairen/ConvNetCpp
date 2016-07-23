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

}

#endif //CONVNETCPP_TESTSOFTMAXLAYER_H
