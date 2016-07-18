//
// Created by kanairen on 2016/07/04.
//

#ifndef CONVNETCPP_MAXPOOLLAYER_H
#define CONVNETCPP_MAXPOOLLAYER_H

#include "GridLayer.h"
#include "activation.h"

class MaxPoolLayer2d : public GridLayer2d {

public:

    MaxPoolLayer2d(unsigned int n_data,
                   unsigned int input_width, unsigned int input_height,
                   unsigned int c,
                   unsigned int kw, unsigned int kh,
                   unsigned int px, unsigned int py)
            : GridLayer2d(n_data, input_width, input_height, c, c, kw,
                          kh, kw, kh, px, py, iden, g_iden, false) { }


    const vector<float> &forward(const vector<float> &input) {

        /*
         * 入力の重み付き和を順伝播する関数
         *
         * inputs : n_in行 n_data列 の入力データ
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        int j_out, i_in;
        int max_idx;
        float p_value, max;
        for (int i_data = 0; i_data < n_data; ++i_data) {

            for (int co = 0; co < c_out; ++co) {
                for (int ci = 0; ci < c_in; ++ci) {
                    for (int y = 0; y < input_height - kh; y += sy) {
                        for (int x = 0; x < input_width - kw; x += sx) {

                            j_out = co * output_height * output_width +
                                    y / sy * output_width +
                                    x / sx;

                            max = -MAXFLOAT;

                            for (int ky = 0; ky < kh; ++ky) {
                                for (int kx = 0; kx < kw; ++kx) {
                                    i_in = ci * input_width * input_height +
                                           y * input_width + x +
                                           ky * input_width + kx;

                                    p_value = input[i_in * n_data + i_data];

                                    if (p_value > max) {
                                        max = p_value;
                                        max_idx = i_in;
                                    }

                                    weights[j_out * n_in + i_in] = 0.f;

                                }
                            }

                            weights[j_out * n_in + max_idx] = 1.f;
                            u[j_out * n_data + i_data] = max;
                            z[j_out * n_data + i_data] = max;

                        }
                    }
                }
            }


        }

#ifdef PROFILE_ENABLED
        std::cout << "MaxPoolLayer2d::forward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

        return z;
    }

    void backward(const vector<float> &next_weights,
                  const vector<float> &next_delta,
                  const vector<float> &prev_output,
                  const unsigned int next_n_out,
                  const float learning_rate) {

        /*
         * 誤差逆伝播で微分導出に用いるデルタを計算する関数
         * プーリングレイヤなので、updateはしない
         *
         * next_weight : 次層重み行列
         * next_delta : 次層デルタ
         * prev_output : 前層の出力
         * learning_rate : 学習率(0≦learning_rate≦1)
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        float d;
        for (int i_data = 0; i_data < n_data; ++i_data) {
            for (int i_out = 0; i_out < n_out; ++i_out) {
                d = 0.f;
                for (int i_n_out = 0; i_n_out < next_n_out; ++i_n_out) {
                    // デルタを導出
                    d += next_weights[i_n_out * n_out + i_out] *
                         next_delta[i_n_out * n_data + i_data];
                }
                // 順伝播の活性化関数が恒等写像なので、活性化関数の導関数は使わない
                delta[i_out * n_data + i_data] = d * u[i_out * n_data + i_data];
            }
        }

#ifdef PROFILE_ENABLED
        std::cout << "MaxPoolLayer2d::backward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
    }


};

#endif //CONVNETCPP_MAXPOOLLAYER_H
