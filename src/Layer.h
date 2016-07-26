//
// Created by kanairen on 2016/06/30.
//

#ifndef CONVNETCPP_ABSTRACTLAYER_H
#define CONVNETCPP_ABSTRACTLAYER_H

#include "config.h"
#include <random>
#include <iostream>
#include <vector>

#include <Eigen/Core>

using std::vector;

using Eigen::MatrixXi;
using Eigen::MatrixXf;
using Eigen::VectorXf;

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

        const int n_i = n_in;
        const int n_o = n_out;
        const int n_d = n_data;
        int i_out, i_in, i_data;

        // W ← W - ε * dw / N　のうち、ε/Nを先に計算してしまう
        const float lr = learning_rate / n_d;

        for (i_out = 0; i_out < n_o; ++i_out) {
            db = 0.f;
            for (i_in = 0; i_in < n_i; ++i_in) {
                dw = 0.f;
                for (i_data = 0; i_data < n_d; ++i_data) {
                    d = delta[i_out * n_d + i_data];
                    dw += d * prev_output[i_in * n_d + i_data];
                    if (i_in == 0) {
                        db += d;
                    }
                }
                weights[i_out * n_i + i_in] -= lr * dw;
            }
            biases[i_out] -= lr * db;
        }

#ifdef PROFILE_ENABLED
        std::cout << "Layer::update : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

    }

public:

    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
          float (*activation)(float), float (*grad_activation)(float),
          bool is_weight_init_enabled = true,
          float weight_constant_value = 0.f)
            : n_data(n_data), n_in(n_in), n_out(n_out),
              activation(activation), grad_activation(grad_activation),
              weights(vector<float>(n_out * n_in, weight_constant_value)),
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

        float out, bias;
        const int n_d = n_data;
        const int n_o = n_out;
        const int n_i = n_in;
        int i_data, i_out, i_in, idx_output;
        int out_ni = 0, out_nd = 0;

        for (i_out = 0; i_out < n_o; ++i_out) {
            bias = biases[i_out];
            for (i_data = 0; i_data < n_d; ++i_data) {
                out = bias;
                for (i_in = 0; i_in < n_i; ++i_in) {
                    out += w[out_ni + i_in] * input[i_in * n_d + i_data];
                }
                idx_output = out_nd + i_data;
                u[idx_output] = out;
                z[idx_output] = activation(out);
            }
            out_ni += n_i;
            out_nd += n_d;
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

        const int n_o = n_out;
        const int n_d = n_data;
        int i_n_out, i_out, i_data;
        float next_w;

        std::fill(delta.begin(), delta.end(), 0.f);

        // キャッシュヒット率を上げるため、i_n_outループを一番外側に持ってきている
        for (i_n_out = 0; i_n_out < next_n_out; ++i_n_out) {
            for (i_out = 0; i_out < n_o; ++i_out) {
                next_w = next_weights[i_n_out * n_o + i_out];
                if (next_w != 0) {
                    for (i_data = 0; i_data < n_d; ++i_data) {
                        // デルタを導出
                        delta[i_out * n_d + i_data] +=
                                next_w *
                                next_delta[i_n_out * n_d + i_data] *
                                grad_activation(u[i_out * n_d + i_data]);
                    }
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


class Layer_ {

    /*
     * ニューラルネットワークの全結合クラス
     */

protected:

    unsigned int n_data; // データ数
    unsigned int n_in; // 入力ユニット数
    unsigned int n_out; // 出力ユニット数

    // 重み行列
    MatrixXf weights;

    // バイアスベクトル
    VectorXf biases;

    // Backwardで用いるデルタ
    MatrixXf delta;

    // データごとのForwardの重み付き和行列
    MatrixXf u;

    // uの各要素に活性化関数を適用した行列
    MatrixXf z;

    // 要素が全て１のベクトル
    VectorXf ones_vec;

    // 活性化関数
    float (*activation)(float);

    // 活性化関数微分形
    float (*grad_activation)(float);

    Layer_() = delete;


    virtual void update(const MatrixXf &prev_output,
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

        // W ← W - ε * dw / N　のうち、ε/Nを先に計算してしまう
        const float lr = learning_rate / n_data;

        weights.array() -= (delta * prev_output.transpose()).array() * lr;

        biases -= delta * ones_vec * lr;

#ifdef PROFILE_ENABLED
        std::cout << "Layer::update : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

    }

public:

    Layer_(unsigned int n_data, unsigned int n_in, unsigned int n_out,
           float (*activation)(float), float (*grad_activation)(float),
           bool is_weight_rand_init_enabled = true,
           float weight_constant_value = 0.f)
            : n_data(n_data), n_in(n_in), n_out(n_out),
              activation(activation), grad_activation(grad_activation),
              weights(MatrixXf::Constant(n_out, n_in, weight_constant_value)),
              biases(VectorXf::Zero(n_out)),
              delta(MatrixXf::Zero(n_out, n_data)),
              u(MatrixXf::Zero(n_out, n_data)),
              z(MatrixXf::Zero(n_out, n_data)),
              ones_vec(VectorXf::Ones(n_data)) {

        if (is_weight_rand_init_enabled) {

            // 乱数生成器
            std::random_device rnd;
            std::mt19937 mt(rnd());
            std::uniform_real_distribution<float> uniform(
                    -sqrtf(6.f / (n_in + n_out)),
                    sqrtf(6.f / (n_in + n_out)));

            // 重みパラメタの初期化
            for (int j = 0; j < n_in; ++j) {
                for (int i = 0; i < n_out; ++i) {
                    weights(i, j) = uniform(mt);
                }
            }

        }

    }

    virtual ~Layer_() { };

    virtual const unsigned int get_n_in() const { return n_in; }

    virtual const unsigned int get_n_out() const { return n_out; }

    virtual const MatrixXf &get_delta() const { return delta; }

    virtual void set_delta(const MatrixXf &d) { delta = d; }

    virtual const MatrixXf &get_u() const { return u; }

    virtual const MatrixXf &get_z() const { return z; }

    virtual const MatrixXf &get_weights() { return weights; }

    virtual const void set_weights(const MatrixXf &w) { weights = w; }

    virtual const VectorXf &get_biases() { return biases; }

    virtual const MatrixXf &forward(const MatrixXf &input) {

        /*
         * 入力の重み付き和を順伝播する関数
         *
         * input : n_in行 n_data列 の入力データ
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        const MatrixXf &w = get_weights();

        u = (w * input).colwise() + biases;
        for (int j = 0; j < z.cols(); ++j) {
            for (int i = 0; i < z.rows(); ++i) {
                z(i, j) = activation(u(i, j));
            }
        }

#ifdef PROFILE_ENABLED
        std::cout << "Layer::forward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

        return z;
    }

    virtual void backward(const MatrixXf &next_weights,
                          const MatrixXf &next_delta,
                          const MatrixXf &prev_output,
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

        delta = next_weights.transpose() * next_delta;

        for (int j = 0; j < n_data; ++j) {
            for (int i = 0; i < n_out; ++i) {
                delta(i, j) = delta(i, j) * grad_activation(u(i, j));
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
