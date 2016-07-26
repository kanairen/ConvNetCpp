//
// Created by kanairen on 2016/07/25.
//

#ifndef CONVNETCPP_TESTMAXPOOLLAYER_H
#define CONVNETCPP_TESTMAXPOOLLAYER_H

#include "../src/MaxPoolLayer.h"

namespace max_pool_layer {

    void test_init() {

        std::cout << "TestMaxPoolLayer::test_init()... " << std::endl;

        unsigned int n_data = 1;
        unsigned int input_width = 3;
        unsigned int input_height = 3;
        unsigned int c_in = 1;
        unsigned int kw = 2;
        unsigned int kh = 2;
        unsigned int px = 0;
        unsigned int py = 0;

        MaxPoolLayer2d_ layer(n_data, input_width, input_height, c_in, kw, kh,
                              px, py);

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

        assert(biases.size() == n_out);

        assert(delta.rows() == n_out);
        assert(delta.cols() == n_data);

        assert(z.rows() == n_out);
        assert(z.cols() == n_data);

    }


    void test_forward() {

        std::cout << "TestMaxPoolLayer::test_forward()... " << std::endl;

        unsigned int n_data = 1;
        unsigned int input_width = 4;
        unsigned int input_height = 4;
        unsigned int c_in = 1;
        unsigned int kw = 2;
        unsigned int kh = 2;
        unsigned int px = 0;
        unsigned int py = 0;

        MaxPoolLayer2d_ layer(n_data, input_width, input_height, c_in, kw, kh,
                              px, py);

        const unsigned int n_in = layer.get_n_in();

        MatrixXf input(n_in, n_data);
        input << 1, 5, 9, 13,
                2, 6, 10, 14,
                3, 7, 11, 15,
                4, 8, 12, 16;

        std::cout << "output : " << std::endl;
        std::cout << layer.forward(input) << std::endl;


    }

    void test_backward() {

        std::cout << "TestMaxPoolLayer::test_backward()... " << std::endl;

        unsigned int n_data = 1;
        unsigned int input_width = 4;
        unsigned int input_height = 4;
        unsigned int c_in = 1;
        unsigned int kw = 2;
        unsigned int kh = 2;
        unsigned int px = 0;
        unsigned int py = 0;

        MaxPoolLayer2d_ layer(n_data, input_width, input_height, c_in, kw, kh,
                              px, py);

        unsigned int next_n_out = 4;
        unsigned int n_out = layer.get_n_out();
        unsigned int n_in = layer.get_n_in();

        MatrixXf input(n_in, n_data);
        input << 1, 5, 9, 13,
                2, 6, 10, 14,
                3, 7, 11, 15,
                4, 8, 12, 16;


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
        result_forward << 6,
                14,
                8,
                16;

        assert(layer.forward(input) == result_forward);

        std::cout << "backward : " << std::endl;
        layer.backward(next_weight, next_delta, input, n_out, 1);

        std::cout << "delta : " << std::endl;
        std::cout << layer.get_delta() << std::endl;

        // test for backward
        MatrixXf result_delta(n_out, n_data);
        result_delta << 24,
                56,
                32,
                64;

        assert(layer.get_delta() == result_delta);

        MatrixXf &&res_w = (MatrixXf &&) MatrixXf(n_out, n_in);
        res_w << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

        std::cout << "weights : " << std::endl;
        std::cout << layer.get_weights() << std::endl;

        assert(layer.get_weights() == res_w);

        VectorXf &&result_biases = (VectorXf &&) VectorXf(n_out);
        result_biases << 0, 0, 0, 0;

        std::cout << "biases : " << std::endl;
        std::cout << layer.get_biases() << std::endl;

        assert(layer.get_biases() == result_biases);

        std::cout << "forward : " << std::endl;
        std::cout << layer.forward(input) << std::endl;

    }
};

#endif //CONVNETCPP_TESTMAXPOOLLAYER_H
