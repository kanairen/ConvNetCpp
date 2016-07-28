//
// Created by kanairen on 2016/06/14.
//

#ifndef CONVNETCPP_TESTMODEL_H
#define CONVNETCPP_TESTMODEL_H

#include<gtest/gtest.h>
#include "../src/Model.h"

class ModelTest : public ::testing::Test {
protected:

    static const unsigned int N_DATA;
    static const unsigned int N_IN;
    static const unsigned int N_HIDDEN;
    static const unsigned int N_OUT;

    constexpr static float (*const ACTIVATION)(float) = iden;

    constexpr static float (*const GRAD_ACTIVATION)(float) = g_iden;

    Layer_ *l1;
    Layer_ *l2;
    vector<Layer_ *> *layers;
    Model_ *model;

    ModelTest() {

        l1 = new Layer_(N_DATA, N_IN, N_HIDDEN, ACTIVATION,
                        GRAD_ACTIVATION, false, 1.f);
        l2 = new SoftMaxLayer_(N_DATA, N_HIDDEN, N_OUT, false, 1.f);

        vector<Layer_ *> *layers = new vector<Layer_ *>{l1, l2};

        model = new Model_(*layers, N_DATA);

    }

    virtual ~ModelTest() {
        delete l1;
        delete l2;
        delete layers;
        delete model;
    }

    virtual void SetUp() {
        std::cout << "ModelTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "ModelTest::TearDown()" << std::endl;
    }

};

const unsigned int ModelTest::N_DATA = 2;
const unsigned int ModelTest::N_IN = 3;
const unsigned int ModelTest::N_HIDDEN = 4;
const unsigned int ModelTest::N_OUT = 5;

TEST_F(ModelTest, test_forward) {

    std::cout << "ModelTest::test_forward()... " << std::endl;

    MatrixXf inputs(N_IN, N_DATA);
    inputs << 1, 4,
            2, 5,
            3, 6;

    const MatrixXf &output = model->forward(inputs);

    std::cout << "output : " << std::endl;
    std::cout << output << std::endl;

    MatrixXf result_output(N_OUT, N_DATA);
    result_output << 0.2, 0.2,
            0.2, 0.2,
            0.2, 0.2,
            0.2, 0.2,
            0.2, 0.2;

    ASSERT_EQ(output, result_output);

}

TEST_F(ModelTest, test_backward) {

    std::cout << "ModelTest::test_backward()... " << std::endl;

    MatrixXf inputs(N_IN, N_DATA);
    inputs << 1, 4,
            2, 5,
            3, 6;

    const MatrixXf &output = model->forward(inputs);

    std::cout << "output : " << std::endl;
    std::cout << output << std::endl;

    MatrixXf result_output(N_OUT, N_DATA);
    result_output << 0.2, 0.2,
            0.2, 0.2,
            0.2, 0.2,
            0.2, 0.2,
            0.2, 0.2;

    ASSERT_EQ(output, result_output);

    MatrixXf &&last_delta = MatrixXf::Ones(N_OUT, N_DATA);

    model->backward(inputs, last_delta, 1);

    //TODO detailed backward test

}

#endif //CONVNETCPP_TESTMODEL_H
