//
// Created by kanairen on 2016/06/13.
//

#ifndef CONVNETCPP_LAYER_H
#define CONVNETCPP_LAYER_H

#include <random>
#include <vector>
#include <iostream>
#include "AbstractLayer.h"

using std::vector;
using std::cout;
using std::endl;

class Layer : public AbstractLayer {
private:

    vector<vector<float>> weights;

    void update(const vector<vector<float>> &prev_output,
                const float learning_rate);

public:
    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
          float (*activation)(float), float (*grad_activation)(float));

    ~Layer() { }

    const unsigned int get_n_out() const { return n_out; }

    const vector<vector<float>> &get_weights() const { return weights; }

    const vector<vector<float>> &get_delta() const { return delta; }

    const vector<vector<float>> &get_z() const { return z; };

    const vector<vector<float>> &forward(const vector<vector<float>> &input);

    void backward(const vector<vector<float>> &last_delta,
                  const vector<vector<float>> &prev_output,
                  const float learning_rate);

    void backward(const AbstractLayer &next,
                  const vector<vector<float>> &prev_output,
                  const float learning_rate);
};

#endif //CONVNETCPP_LAYER_H
