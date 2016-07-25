//
// Created by kanairen on 2016/06/14.
//

#ifndef CONVNETCPP_TESTMODEL_H
#define CONVNETCPP_TESTMODEL_H

#include "../src/Model.h"

namespace model {
    void test_init() {

        std::cout << "TestModel::test_init()... " << std::endl;

        unsigned int n_data = 2;
        unsigned int n_in = 3;
        unsigned int n_hidden = 4;
        unsigned int n_out = 5;
        float (*activation)(float) = iden;
        float (*grad_activation)(float) = g_iden;


        Layer_ *l1 = new Layer_(n_data, n_in, n_hidden, iden, g_iden, false,
                                1.f);
        Layer_ *l2 = new Layer_(n_data, n_hidden, n_out, iden, g_iden, false,
                                1.f);
        vector<Layer_ *> layers = {l1, l2};

        Model_ model(layers, n_data);

    }

    void test_forward() {

        std::cout << "TestModel::test_forward()... " << std::endl;

        unsigned int n_data = 2;
        unsigned int n_in = 3;
        unsigned int n_hidden = 4;
        unsigned int n_out = 5;
        float (*activation)(float) = iden;
        float (*grad_activation)(float) = g_iden;

        Layer_ *l1 = new Layer_(n_data, n_in, n_hidden, iden, g_iden, false,
                                1.f);
        Layer_ *l2 = new Layer_(n_data, n_hidden, n_out, iden, g_iden, false,
                                1.f);
        vector<Layer_ *> layers = {l1, l2};

        Model_ model(layers, n_data);

        MatrixXf inputs(n_in, n_data);
        inputs << 1, 4,
                2, 5,
                3, 6;

        auto output = model.forward(inputs);

        std::cout << "output : " << std::endl;
        std::cout << output << std::endl;

        MatrixXf result_output(n_out, n_data);
        result_output << 24, 60,
                24, 60,
                24, 60,
                24, 60,
                24, 60;

        assert(output == result_output);

    }

    void test_backward() {

        std::cout << "TestModel::test_backward()... " << std::endl;

        unsigned int n_data = 2;
        unsigned int n_in = 3;
        unsigned int n_hidden = 4;
        unsigned int n_out = 5;
        float (*activation)(float) = iden;
        float (*grad_activation)(float) = g_iden;

        Layer_ *l1 = new Layer_(n_data, n_in, n_hidden, iden, g_iden, false,
                                1.f);
        Layer_ *l2 = new SoftMaxLayer_(n_data, n_hidden, n_out, false, 1.f);

        vector<Layer_ *> layers = {l1, l2};

        Model_ model(layers, n_data);

        MatrixXf inputs(n_in, n_data);
        inputs << 1, 4,
                2, 5,
                3, 6;

        auto output = model.forward(inputs);

        std::cout << "output : " << std::endl;
        std::cout << output << std::endl;

        MatrixXf result_output(n_out, n_data);
        result_output << 0.2, 0.2,
                0.2, 0.2,
                0.2, 0.2,
                0.2, 0.2,
                0.2, 0.2;

        assert(output == result_output);


        MatrixXf &&last_delta = MatrixXf::Ones(n_out, n_data);

        model.backward(inputs, last_delta, 1);

        //TODO detailed backward test

    }
}

#endif //CONVNETCPP_TESTMODEL_H
