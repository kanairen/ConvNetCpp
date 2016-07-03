//
// Created by kanairen on 2016/06/30.
//

#ifndef CONVNETCPP_ABSTRACTLAYER_H
#define CONVNETCPP_ABSTRACTLAYER_H


#include <random>
#include <iostream>
#include <vector>

using std::vector;

class Layer {
protected:
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

    Layer() = delete;

public:

    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
                  float (*activation)(float), float (*grad_activation)(float))
            : n_data(n_data), n_in(n_in), n_out(n_out),
              activation(activation), grad_activation(grad_activation),
              weights(vector<vector<float>>(n_out, vector<float>(n_in, 0.f))),
              biases(vector<float>(n_out, 0.f)),
              delta(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
              u(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
              z(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))) {


        // 乱数生成器
        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::uniform_real_distribution<float> uniform(
                -sqrtf(6.f / (n_in + n_out)),
                sqrtf(6.f / (n_in + n_out)));

        // 重みパラメタの初期化
        for (vector<float> &row : weights) {
            for (float &w : row) {
                w = uniform(mt);
            }
        }

    }


    virtual ~Layer() { };

    virtual const unsigned int get_n_out() const { return n_out; }

    virtual const vector<vector<float>> &get_delta() const { return delta; }

    virtual const vector<vector<float>> &get_z() const { return z; }

    virtual const vector<vector<float>> &get_weights() { return weights; }

    const vector<vector<float>> &forward(const vector<vector<float>> &input) {
        const vector<vector<float>> &w = get_weights();
        float out;
        for (int i_data = 0; i_data < n_data; ++i_data) {
            for (int i_out = 0; i_out < n_out; ++i_out) {
                out = 0.f;
                for (int i_in = 0; i_in < n_in; ++i_in) {
                    out += w[i_out][i_in] * input[i_in][i_data];
                }
                out += biases[i_out];
                u[i_out][i_data] = out;
                z[i_out][i_data] = activation(out);
            }
        }
        return z;
    }

    void backward(const vector<vector<float>> &last_delta,
                  const vector<vector<float>> &prev_output,
                  const float learning_rate) {

#ifdef DEBUG_LAYER
        if (last_delta.size() != delta.size() &&
            last_delta[0].size() != delta[0].size()) {
            std::cerr << "Layer::backward : size of delta is not correct.";
            exit(1);
        }
#endif

        // 出力層のデルタとしてコピー
        unsigned long last_delta_length = last_delta[0].size();
        for (int i = 0; i < last_delta.size(); ++i) {
            for (int j = 0; j < last_delta_length; ++j) {
                delta[i][j] = last_delta[i][j];
            }
        }

        // パラメタ更新
        update(prev_output, learning_rate);

    }


    void backward(const vector<vector<float>> &next_weights,
                  const vector<vector<float>> &next_delta,
                  const vector<vector<float>> &prev_output,
                  const float learning_rate) {
        unsigned long next_n_out = next_weights.size();
        float d;
        for (int i_data = 0; i_data < n_data; ++i_data) {
            for (int i_out = 0; i_out < n_out; ++i_out) {
                d = 0.f;
                for (int i_n_out = 0; i_n_out < next_n_out; ++i_n_out) {
                    // デルタを導出
                    d += next_weights[i_n_out][i_out] *
                         next_delta[i_n_out][i_data];
                }
                delta[i_out][i_data] = d * grad_activation(u[i_out][i_data]);
            }
        }

        // パラメタ更新
        update(prev_output, learning_rate);

    }


    virtual void update(const vector<vector<float>> &prev_output,
                        const float learning_rate) {
        float dw, db;

        for (int i_out = 0; i_out < n_out; ++i_out) {
            for (int i_in = 0; i_in < n_in; ++i_in) {
                dw = 0.f;
                db = 0.f;
                for (int i_data = 0; i_data < n_data; ++i_data) {
                    // オーバフローを防ぐため、先に学習率を掛ける
                    dw += learning_rate * delta[i_out][i_data] *
                          prev_output[i_in][i_data];
                    db += learning_rate * delta[i_out][i_data];

                }
                weights[i_out][i_in] -= (dw / n_data);
            }
            biases[i_out] -= (db / n_data);
        }

    }

};

#endif //CONVNETCPP_ABSTRACTLAYER_H
