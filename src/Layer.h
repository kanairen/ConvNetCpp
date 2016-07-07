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

    /*
     * ニューラルネットワークの全結合クラス
     */

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

    virtual void update(const vector<vector<float>> &prev_output,
                        const float learning_rate) {

        /*
         * 学習パラメタの更新を行う関数
         *
         * prev_output : 前層の出力
         * learning_rate : 学習率(0≦learning_rate≦1)
         */

        float dw, db;

        for (int i_out = 0; i_out < n_out; ++i_out) {
            for (int i_in = 0; i_in < n_in; ++i_in) {
                dw = 0.f;
                db = 0.f;
                for (int i_data = 0; i_data < n_data; ++i_data) {
                    dw += delta[i_out][i_data] * prev_output[i_in][i_data];
                    db += delta[i_out][i_data];
                }
                weights[i_out][i_in] -= learning_rate * (dw / n_data);
            }
            biases[i_out] -= learning_rate * (db / n_data);
        }

    }

public:

    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
          float (*activation)(float), float (*grad_activation)(float),
          bool is_weight_init_enabled = true)
            : n_data(n_data), n_in(n_in), n_out(n_out),
              activation(activation), grad_activation(grad_activation),
              weights(vector<vector<float>>(n_out, vector<float>(n_in, 0.f))),
              biases(vector<float>(n_out, 0.f)),
              delta(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
              u(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
              z(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))) {

        if (is_weight_init_enabled) {

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

    }

    virtual ~Layer() { };

    virtual const unsigned int get_n_out() const { return n_out; }

    virtual const vector<vector<float>> &get_delta() const { return delta; }

    virtual const vector<vector<float>> &get_z() const { return z; }

    virtual const vector<vector<float>> &get_weights() { return weights; }

    virtual const vector<vector<float>> &forward(
            const vector<vector<float>> &input) {

        /*
         * 入力の重み付き和を順伝播する関数
         *
         * inputs : n_in行 n_data列 の入力データ
         */

        const vector<vector<float>> &w = get_weights();
        float out;
        for (int i_data = 0; i_data < n_data; ++i_data) {
            for (int i_out = 0; i_out < n_out; ++i_out) {
                out = biases[i_out];
                for (int i_in = 0; i_in < n_in; ++i_in) {
                    out += w[i_out][i_in] * input[i_in][i_data];
                }
                u[i_out][i_data] = out;
                z[i_out][i_data] = activation(out);
            }
        }

        return z;
    }


    virtual void backward(const vector<vector<float>> &next_weights,
                          const vector<vector<float>> &next_delta,
                          const vector<vector<float>> &prev_output,
                          const float learning_rate) {

        /*
         * 誤差逆伝播で微分導出に用いるデルタを計算する関数
         *
         * next_weight : 次層重み行列
         * next_delta : 次層デルタ
         * prev_output : 前層の出力
         * learning_rate : 学習率(0≦learning_rate≦1)
         */

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


};

#endif //CONVNETCPP_ABSTRACTLAYER_H
