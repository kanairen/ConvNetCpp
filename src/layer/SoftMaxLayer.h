//
// Created by kanairen on 2016/07/06.
//

#ifndef CONVNETCPP_SOFTMAXLAYER_H
#define CONVNETCPP_SOFTMAXLAYER_H

#include <float.h>
#include "Layer.h"
#include "../activation.h"

class SoftMaxLayer_ : public Layer_ {

public:
    SoftMaxLayer_(unsigned int n_data, unsigned int n_in, unsigned int n_out,
                  bool is_weight_rand_init_enabled = true,
                  float weight_constant_value = 0.f,
                  bool is_dropout_enabled = false,
                  float dropout_rate = 0.5f)
            : Layer_(n_data, n_in, n_out, iden, g_iden,
                     is_weight_rand_init_enabled, weight_constant_value,
                     is_dropout_enabled, dropout_rate) { }

    ~SoftMaxLayer_() { }

    const MatrixXf &forward(const MatrixXf &input, bool is_train) {

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

        const VectorXf &&max_u = u.colwise().maxCoeff();

        for (int j = 0; j < z.cols(); ++j) {
            for (int i = 0; i < z.rows(); ++i) {
                z(i, j) = expf(u(i, j) - max_u[j]);
            }
        }

        auto sum_z = z.colwise().sum();

        for (int j = 0; j < z.cols(); ++j) {
            z.col(j) /= sum_z[j];
        }

        // Dropout
        if (is_dropout_enabled) {
            dropout(is_train);
        }

#ifdef PROFILE_ENABLED
        std::cout << "SoftMaxLayer::forward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

        return z;

    }


    void backward(const MatrixXf &next_weights,
                  const MatrixXf &next_delta,
                  const MatrixXf &prev_output,
                  const unsigned int next_n_out,
                  const float learning_rate) {

        /*
         * 誤差逆伝播で微分導出に用いるデルタを計算する関数
         * 出力層用
         *
         * last_delta : 出力層デルタ
         * prev_output : 前層の出力
         * learning_rate : 学習率
         */
#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        // 出力層のデルタとしてコピー
        delta = next_delta;

        // Dropout
        if (is_dropout_enabled) {
            delta = delta.array() * dropout_filter.array();
        }

        // パラメタ更新
        update(prev_output, learning_rate);

#ifdef PROFILE_ENABLED
        std::cout << "SoftMaxLayer::backward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
    }

};

#endif //CONVNETCPP_SOFTMAXLAYER_H
