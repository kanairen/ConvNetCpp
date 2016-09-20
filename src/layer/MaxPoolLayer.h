//
// Created by kanairen on 2016/07/04.
//

#ifndef CONVNETCPP_MAXPOOLLAYER_H
#define CONVNETCPP_MAXPOOLLAYER_H

#include "GridLayer.h"
#include "../activation.h"


class MaxPoolLayer2d_ : public GridLayer2d_ {

public:

    MaxPoolLayer2d_(unsigned int n_data,
                    unsigned int input_width, unsigned int input_height,
                    unsigned int c,
                    unsigned int kw, unsigned int kh,
                    unsigned int px, unsigned int py)
            : GridLayer2d_(n_data, input_width, input_height, c, c, kw,
                           kh, kw, kh, px, py, iden, g_iden, false) { }


    const MatrixXf &forward(const MatrixXf &input) {

        /*
         * 入力の重み付き和を順伝播する関数
         *
         * inputs : n_in行 n_data列 の入力データ
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        weights.setZero();

        int j_out, i_in;
        int max_idx;
        float p_value, max;
        int count = 0;
        for (int i_data = 0; i_data < n_data; ++i_data) {

            for (int co = 0; co < c_out; ++co) {
                for (int y = 0; y <= input_height - kh; y += sy) {
                    for (int x = 0; x <= input_width - kw; x += sx) {

                        j_out = co * output_height * output_width +
                                y / sy * output_width +
                                x / sx;

                        max = -MAXFLOAT;

                        for (int ci = 0; ci < c_in; ++ci) {
                            for (int ky = 0; ky < kh; ++ky) {
                                for (int kx = 0; kx < kw; ++kx) {
                                    i_in = ci * input_width * input_height +
                                           y * input_width + x +
                                           ky * input_width + kx;

                                    p_value = input(i_in, i_data);

                                    if (p_value > max) {
                                        max = p_value;
                                        max_idx = i_in;
                                    }

                                }
                            }
                        }

                        count++;
                        weights(j_out, max_idx) = 1.f;
                        u(j_out, i_data) = max;
                        z(j_out, i_data) = max;

                    }
                }
            }

        }

        // 複数のデータ入力を受け付けるので、非ゼロ重みは　1.f/発火したユニット数　とする
        weights /= weights.sum();

#ifdef PROFILE_ENABLED
        std::cout << "MaxPoolLayer2d::forward : " <<
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

        delta = next_weights.transpose() * next_delta;


#ifdef PROFILE_ENABLED
        std::cout << "MaxPoolLayer2d::backward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
    }


};

#endif //CONVNETCPP_MAXPOOLLAYER_H
