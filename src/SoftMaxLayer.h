//
// Created by kanairen on 2016/07/06.
//

#ifndef CONVNETCPP_SOFTMAXLAYER_H
#define CONVNETCPP_SOFTMAXLAYER_H

#include <float.h>
#include "Layer.h"
#include "activation.h"

class SoftMaxLayer : public Layer {
public:
    SoftMaxLayer(unsigned int n_data, unsigned int n_in, unsigned int n_out)
            : Layer(n_data, n_in, n_out, iden, g_iden) { }

    ~SoftMaxLayer() { }


    const vector<float> &forward(const vector<float> &input) {

        /*
         * 入力の重み付き和を順伝播する関数
         *
         * input : n_in行 n_data列 の入力データ
         */
#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif
        float out, max_out, sum_exp;
        for (int i_data = 0; i_data < n_data; ++i_data) {
            sum_exp = 0.f;
            max_out = FLT_MIN;
            for (int i_out = 0; i_out < n_out; ++i_out) {
                out = 0.f;
                for (int i_in = 0; i_in < n_in; ++i_in) {
                    out += weights[i_out * n_in + i_in] *
                           input[i_in * n_data + i_data];
                }
                out += biases[i_out];
                if (out > max_out) {
                    max_out = out;
                }
                u[i_out * n_data + i_data] = out;
            }
            for (int i_out = 0; i_out < n_out; ++i_out) {
                z[i_out * n_data + i_data] = expf(
                        u[i_out * n_data + i_data] - max_out);
                sum_exp += z[i_out * n_data + i_data];
            }
            for (int i_out = 0; i_out < n_out; ++i_out) {
                z[i_out * n_data + i_data] /= sum_exp;
            }
        }
#ifdef PROFILE_ENABLED
        std::cout << "SoftMaxLayer::forward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
        return z;
    }


    void backward(const vector<float> &next_weights,
                  const vector<float> &last_delta,
                  const vector<float> &prev_output,
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
        std::copy(last_delta.begin(), last_delta.end(), delta.begin());

        // パラメタ更新
        update(prev_output, learning_rate);

#ifdef PROFILE_ENABLED
        std::cout << "SoftMaxLayer::backward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
    }


};

#endif //CONVNETCPP_SOFTMAXLAYER_H
