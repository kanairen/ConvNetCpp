//
// Created by 金井廉 on 2016/07/23.
//

#ifndef CONVNETCPP_TESTCONVLAYER_H
#define CONVNETCPP_TESTCONVLAYER_H

#include "../src/ConvLayer.h"
#include "../src/activation.h"

namespace conv_layer {

    void test_init() {

        std::cout << "TestConvLayer::test_init()... " << std::endl;

        unsigned int n_data = 3;
        unsigned int input_width = 3;
        unsigned int input_height = 3;
        unsigned int c_in = 1;
        unsigned int c_out = 2;
        unsigned int kw = 2;
        unsigned int kh = 2;
        unsigned int sx = 1;
        unsigned int sy = 1;
        unsigned int px = 0;
        unsigned int py = 0;
        float (*activation)(float) = iden;
        float (*grad_activation)(float) = g_iden;

        ConvLayer2d_ layer(n_data, input_width, input_height, c_in, c_out, kw,
                           kh, sx, sy, px, py, activation, grad_activation);

        const MatrixXf &weights = layer.get_weights();
        const VectorXf &biases = layer.get_biases();
        const MatrixXf &delta = layer.get_delta();
        const MatrixXf &z = layer.get_z();

        const unsigned int n_in = layer.get_n_in();
        const unsigned int n_out = layer.get_n_out();

        std::cout << "weights : " << std::endl;
        std::cout << weights << std::endl;

        std::cout << "biases : " << std::endl;
        std::cout << layer.get_biases() << std::endl;

        std::cout << "delta : " << std::endl;
        std::cout << layer.get_delta() << std::endl;

        std::cout << "z : " << std::endl;
        std::cout << layer.get_z() << std::endl;

        assert(weights.rows() == n_out);
        assert(weights.cols() == n_in);

        const auto weight_sum = weights.rowwise().sum();
        bool is_nonzero_exists;
        for (int i = 0; i < weights.rows(); ++i) {
            is_nonzero_exists = false;
            for (int j = 0; j < weights.cols(); ++j) {
                // TODO 比較方法を修正すべき
                if (weights(i, j) != 0.f) {
                    is_nonzero_exists = true;
                }
            }
            assert(is_nonzero_exists);
        }

        assert(biases.size() == n_out);

        assert(delta.rows() == n_out);
        assert(delta.cols() == n_data);

        assert(z.rows() == n_out);
        assert(z.cols() == n_data);

    }


    void test_backward() {

        std::cout << "TestConvLayer::test_backward()... " << std::endl;

        unsigned int n_data = 1;
        unsigned int input_width = 3;
        unsigned int input_height = 3;
        unsigned int c_in = 1;
        unsigned int c_out = 2;
        unsigned int kw = 2;
        unsigned int kh = 2;
        unsigned int sx = 1;
        unsigned int sy = 1;
        unsigned int px = 0;
        unsigned int py = 0;
        float (*activation)(float) = iden;
        float (*grad_activation)(float) = g_iden;

        ConvLayer2d_ layer(n_data, input_width, input_height, c_in, c_out, kw,
                           kh, sx, sy, px, py, activation, grad_activation,
                           false, 1.f);

        unsigned int next_n_out = 4;
        unsigned int n_out = layer.get_n_out();
        unsigned int n_in = layer.get_n_in();

        MatrixXf input(input_width * input_height, 1);
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
        std::cout << layer.get_weights() << std::endl;

        MatrixXf &&next_weight = (MatrixXf &&) MatrixXf::Ones(next_n_out,
                                                              n_out);
        MatrixXf &&next_delta = (MatrixXf &&) MatrixXf::Ones(next_n_out,
                                                             n_data);

        std::cout << "forward : " << std::endl;
        std::cout << layer.forward(input) << std::endl;

        // test for forward
        MatrixXf result_forward(n_out, n_data);
        result_forward <<
        12,
                16,
                24,
                28,

                12,
                16,
                24,
                28;

        assert(layer.forward(input) == result_forward);

        std::cout << "backward : " << std::endl;
        layer.backward(next_weight, next_delta, input, n_out, 1);

        std::cout << "delta : " << std::endl;
        std::cout << layer.get_delta() << std::endl;

        // test for backward
        MatrixXf &&result_delta = (MatrixXf &&) MatrixXf::Constant(n_out,
                                                                   n_data,
                                                                   4);

        assert(layer.get_delta() == result_delta);

        MatrixXf &&res_w = (MatrixXf &&) MatrixXf(n_out, n_in);
        res_w << -47, -63, 0, -95, -111, 0, 0, 0, 0,
                0, -47, -63, 0, -95, -111, 0, 0, 0,
                0, 0, 0, -47, -63, 0, -95, -111, 0,
                0, 0, 0, 0, -47, -63, 0, -95, -111,
                -47, -63, 0, -95, -111, 0, 0, 0, 0,
                0, -47, -63, 0, -95, -111, 0, 0, 0,
                0, 0, 0, -47, -63, 0, -95, -111, 0,
                0, 0, 0, 0, -47, -63, 0, -95, -111;

        std::cout << "weights : " << std::endl;
        std::cout << layer.get_weights() << std::endl;

        assert(layer.get_weights() == res_w);

        VectorXf &&result_biases = (VectorXf &&) VectorXf(n_out);
        result_biases << -4, -4, -4, -4, -4, -4, -4, -4;

        std::cout << "biases : " << std::endl;
        std::cout << layer.get_biases() << std::endl;

        assert(layer.get_biases() == result_biases);

        std::cout << "forward : " << std::endl;
        std::cout << layer.forward(input) << std::endl;

    }

}

#endif //CONVNETCPP_TESTCONVLAYER_H
