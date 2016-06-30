//
// Created by 金井廉 on 2016/06/30.
//

#ifndef CONVNETCPP_ABSTRACTLAYER_H
#define CONVNETCPP_ABSTRACTLAYER_H

#include <vector>

using std::vector;

class AbstractLayer {
protected:
    unsigned int n_data;
    unsigned int n_in;
    unsigned int n_out;

    vector<float> biases;
    vector<vector<float>> delta;

    vector<vector<float>> u;
    vector<vector<float>> z;

    float (*activation)(float);

    float (*grad_activation)(float);

    AbstractLayer() = delete;

    AbstractLayer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
                  float (*activation)(float), float (*grad_activation)(float))
            : n_data(n_data), n_in(n_in), n_out(n_out),
              activation(activation), grad_activation(grad_activation),
              biases(vector<float>(n_out, 0.f)),
              delta(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
              u(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
              z(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))) { };


    virtual void update(const vector<vector<float>> &prev_output,
                        const float learning_rate) = 0;

public:

    virtual ~AbstractLayer(){};

    virtual const unsigned int get_n_out() const = 0;

    virtual const vector<vector<float>> &get_delta() const = 0;

    virtual const vector<vector<float>> &get_z() const = 0;

    virtual const vector<vector<float>> &get_weights() const = 0;

    virtual const vector<vector<float>> &forward(
            const vector<vector<float>> &input) = 0;

    virtual void backward(const vector<vector<float>> &last_delta,
                          const vector<vector<float>> &prev_output,
                          const float learning_rate) = 0;

    virtual void backward(const AbstractLayer &next,
                          const vector<vector<float>> &prev_output,
                          const float learning_rate) = 0;
};

#endif //CONVNETCPP_ABSTRACTLAYER_H
