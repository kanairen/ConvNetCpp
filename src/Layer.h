//
// Created by kanairen on 2016/06/13.
//

#ifndef CONVNETCPP_LAYER_H
#define CONVNETCPP_LAYER_H

#include <vector>
#include <iostream>
#include <exception>

using std::vector;
using std::cout;
using std::endl;

class Layer {
private:
    unsigned int n_data;
    unsigned int n_in;
    unsigned int n_out;
    vector<vector<float>> weights;
    vector<float> biases;
    vector<vector<float>> delta;

    vector<vector<float>> out_forward;

    float (*activation)(float);

public:
    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
          float (*activation)(float)) :
            n_data(n_data), n_in(n_in), n_out(n_out), activation(activation),
            weights(vector<vector<float>>(n_out, vector<float>(n_in, 0.))),
            biases(vector<float>(n_out, 0.)),
            delta(vector<vector<float>>(n_out, vector<float>(n_data, 0.))),
            out_forward(
                    vector<vector<float>>(n_out, vector<float>(n_data, 0.))) {
    }

    ~Layer() { }

    const vector<vector<float>> &forward(vector<vector<float>> &input);

    void backward(Layer &next);
};

#endif //CONVNETCPP_LAYER_H
