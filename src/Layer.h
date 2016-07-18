//
// Created by kanairen on 2016/06/30.
//

#ifndef CONVNETCPP_ABSTRACTLAYER_H
#define CONVNETCPP_ABSTRACTLAYER_H

#include "config.h"
#include <random>
#include <iostream>
#include <vector>

using std::vector;

class Layer {

    /*
     * ニューラルネットワークの全結合クラス
     */

protected:
    unsigned int n_data; // データ数
    unsigned int n_in; // 入力ユニット数
    unsigned int n_out; // 出力ユニット数

    // 重み行列
    vector<float> weights;

    // バイアスベクトル
    vector<float> biases;

    // Backwardで用いるデルタ
    vector<float> delta;

    // データごとのForwardの重み付き和行列
    vector<float> u;

    // uの各要素に活性化関数を適用した行列
    vector<float> z;

    // 活性化関数
    float (*activation)(float);

    // 活性化関数微分形
    float (*grad_activation)(float);

    Layer() = delete;

    virtual void update(const vector<float> &prev_output,
                        const float learning_rate) {

        /*
         * 学習パラメタの更新を行う関数
         *
         * prev_output : 前層の出力
         * learning_rate : 学習率(0≦learning_rate≦1)
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif
        float dw, db, d;

        for (int i_out = 0; i_out < n_out; ++i_out) {
            db = 0.f;
            for (int i_in = 0; i_in < n_in; ++i_in) {
                dw = 0.f;
                for (int i_data = 0; i_data < n_data; ++i_data) {
                    d = delta[i_out * n_data + i_data];
                    dw += d * prev_output[i_in * n_data + i_data];
                    if (i_in == n_in - 1) {
                        db += d;
                    }
                }
                weights[i_out * n_in + i_in] -= learning_rate * (dw / n_data);
            }
            biases[i_out] -= learning_rate * (db / n_data);
        }
#ifdef PROFILE_ENABLED
        std::cout << "Layer::update : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
    }

public:

    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
          float (*activation)(float), float (*grad_activation)(float),
          bool is_weight_init_enabled = true)
            : n_data(n_data), n_in(n_in), n_out(n_out),
              activation(activation), grad_activation(grad_activation),
              weights(vector<float>(n_out * n_in, 0.f)),
              biases(vector<float>(n_out, 0.f)),
              delta(vector<float>(n_out * n_data, 0.f)),
              u(vector<float>(n_out * n_data, 0.f)),
              z(vector<float>(n_out * n_data, 0.f)) {

        if (is_weight_init_enabled) {

            // 乱数生成器
            std::random_device rnd;
            std::mt19937 mt(rnd());
            std::uniform_real_distribution<float> uniform(
                    -sqrtf(6.f / (n_in + n_out)),
                    sqrtf(6.f / (n_in + n_out)));

            // 重みパラメタの初期化
            for (float &w : weights) {
                w = uniform(mt);
            }

        }

    }

    virtual ~Layer() { };

    virtual const unsigned int get_n_out() const { return n_out; }

    virtual const vector<float> &get_delta() const { return delta; }

    virtual const vector<float> &get_z() const { return z; }

    virtual const vector<float> &get_weights() { return weights; }

    virtual const vector<float> &forward(const vector<float> &input) {

        /*
         * 入力の重み付き和を順伝播する関数
         *
         * input : n_in行 n_data列 の入力データ
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        const vector<float> &w = get_weights();

        float out;
        unsigned int idx;
        for (int i_data = 0; i_data < n_data; ++i_data) {
            for (int i_out = 0; i_out < n_out; ++i_out) {
                out = biases[i_out];
                for (int i_in = 0; i_in < n_in; ++i_in) {
                    out += w[i_out * n_in + i_in] *
                           input[i_in * n_data + i_data];
                }
                u[i_out * n_data + i_data] = out;
                z[i_out * n_data + i_data] = activation(out);
            }
        }

#ifdef PROFILE_ENABLED
        std::cout << "Layer::forward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

        return z;
    }


    virtual void backward(const vector<float> &next_weights,
                          const vector<float> &next_delta,
                          const vector<float> &prev_output,
                          const unsigned int next_n_out,
                          const float learning_rate) {

        /*
         * 誤差逆伝播で微分導出に用いるデルタを計算する関数
         *
         * next_weight : 次層重み行列
         * next_delta : 次層デルタ
         * prev_output : 前層の出力
         * learning_rate : 学習率(0≦learning_rate≦1)
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        std::fill(delta.begin(), delta.end(), 0.f);

        // キャッシュヒット率を上げるため、i_n_outループを一番外側に持ってきている
        for (int i_n_out = 0; i_n_out < next_n_out; ++i_n_out) {
            for (int i_out = 0; i_out < n_out; ++i_out) {
                for (int i_data = 0; i_data < n_data; ++i_data) {
                    // デルタを導出
                    delta[i_out * n_data + i_data] +=
                            next_weights[i_n_out * n_out + i_out] *
                            next_delta[i_n_out * n_data + i_data] *
                            grad_activation(u[i_out * n_data + i_data]);
                }
            }
        }

        // パラメタ更新
        update(prev_output, learning_rate);

#ifdef PROFILE_ENABLED
        std::cout << "Layer::backward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

    }


};

#endif //CONVNETCPP_ABSTRACTLAYER_H
