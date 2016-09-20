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
