//
// Created by kanairen on 2016/06/14.
//

#ifndef CONVNETCPP_MODEL_H
#define CONVNETCPP_MODEL_H

#include <cmath>
#include <vector>
#include "Layer.h"

using Eigen::VectorXi;
using std::vector;

class Model {
private:
    vector<Layer *> &layers;
    vector<float> out_forward;

    Model() = delete;

public:
    Model(vector<Layer *> &layers, unsigned int n_data) :
            layers(layers),
            out_forward(vector<float>(layers.back()->get_n_out() * n_data,
                                      0.f)) { };

    ~Model() { };

    const vector<float> &forward(const vector<float> &inputs);

    void backward(const vector<float> &inputs,
                  const vector<float> &last_delta, float learning_rate);

    static void argmax(const vector<float> &y, vector<int> &predict,
                       const unsigned int n_out, const unsigned int n_data);

    static float error(const vector<int> &predict, const vector<int> &answer);


};


#endif //CONVNETCPP_MODEL_H
