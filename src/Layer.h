//
// Created by kanairen on 2016/06/13.
//

#ifndef CONVNETCPP_LAYER_H
#define CONVNETCPP_LAYER_H

//#define SHOW_DW
//#define SHOW_DELTA
//#define SHOW_WEIGHT_INIT

#include <random>
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

    vector<vector<float>> u;
    vector<vector<float>> z;

    float (*activation)(float);

    float (*grad_activation)(float);

    void update(const vector<vector<float>> &prev_output,
                const float learning_rate);

public:
    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
          float (*activation)(float), float (*grad_activation)(float));

    ~Layer() { }

    const unsigned int get_n_out() const { return n_out; }

    const vector<vector<float>> &get_z() { return z; }

    const vector<vector<float>> &forward(const vector<vector<float>> &input);

    void backward(const vector<vector<float>> &last_delta,
                  const vector<vector<float>> &prev_output,
                  const float learning_rate);

    void backward(const Layer &next,
                  const vector<vector<float>> &prev_output,
                  const float learning_rate);
};

#endif //CONVNETCPP_LAYER_H
