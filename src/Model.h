//
// Created by kanairen on 2016/06/14.
//

#ifndef CONVNETCPP_MODEL_H
#define CONVNETCPP_MODEL_H

#include <cmath>
#include <vector>
#include "Layer.h"

using std::vector;

class Model {
private:
    vector<Layer> &layers;
    vector<vector<float>> out_forward;
public:
    Model(vector<Layer> &layers, unsigned int n_data) :
            layers(layers),
            out_forward(vector<vector<float>>(layers.back().get_n_out(),
                                              vector<float>(n_data, 0.f))) { };

    ~Model() { };

    const vector<vector<float>> &forward(vector<vector<float>> &inputs);

    void backward(const vector<vector<float>> &inputs,
                  const vector<vector<float>> &last_delta, float learning_rate);

    static void softmax(const vector<vector<float>> &outputs,
                        vector<vector<float>> &y);

    static void argmax(const vector<vector<float>> &y,
                       vector<int> &predict);

    static float error(const vector<int> &predict, const vector<int> &answer);


};


#endif //CONVNETCPP_MODEL_H
