//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_CONVLAYER_H
#define CONVNETCPP_CONVLAYER_H

#include <random>
#import "GridLayer.h"

namespace ConvLayerConst {
    static const int T_WEIGHT_DISABLED = -1;
}

class ConvLayer2d : public GridLayer2d {

    /*
     * 入力を二次元画像とする、ニューラルネットワークの畳み込み層クラス
     */

private:

    // 成分数 (フィルタ幅)^2 × (入力チャネル) × （出力チャネル） のフィルタベクトル
    vector<float> h;

    // 入力画素(i, j)と重なるフィルタのインデックスを保持するベクトル
    vector<vector<int>> t;

    void update(const vector<vector<float>> &prev_output,
                const float learning_rate) {

        /*
         * フィルタ重み・バイアス値を更新する
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
                if (t[i_out][i_in] != ConvLayerConst::T_WEIGHT_DISABLED) {
                    h[t[i_out][i_in]] -= learning_rate * dw / n_data;
                }
            }
            biases[i_out] -= learning_rate * db / n_data;
        }

    }

public:

    ConvLayer2d(unsigned int n_data,
                unsigned int input_width, unsigned int input_height,
                unsigned int c_in, unsigned int c_out,
                unsigned int kw, unsigned int kh, unsigned int stride,
                float (*activation)(float), float (*grad_activation)(float))
            : GridLayer2d(n_data, input_width, input_height, c_in, c_out, kw,
                          kh, stride, activation, grad_activation),
              h(vector<float>(kw * kh * c_in * c_out)),
              t(vector<vector<int>>(n_out, vector<int>(
                      n_in, ConvLayerConst::T_WEIGHT_DISABLED))) {

        // 乱数生成器
        std::random_device rnd;
        std::mt19937 mt(rnd());
        int f_in = c_in * kw * kh;
        int f_out = c_out * kw * kh;
        std::uniform_real_distribution<float> uniform(
                -sqrtf(6.f / (f_in + f_out)),
                sqrtf(6.f / (f_in + f_out)));

        // filter要素を初期化
        for (float &f : h) {
            f = uniform(mt);
        }

        // tを初期化
        int j_out, i_in;
        unsigned int output_width = filter_outsize(input_width, kw, stride, 0,
                                                   false);
        unsigned int output_height = filter_outsize(input_height, kh, stride, 0,
                                                    false);
        for (int co = 0; co < c_out; ++co) {
            for (int ci = 0; ci < c_in; ++ci) {
                for (int y = 0; y < input_height - kh; y += stride) {
                    for (int x = 0; x < input_width - kw; x += stride) {

                        j_out = co * output_height * output_width +
                                y / stride * output_width +
                                x / stride;

                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                i_in = ci * input_width * input_height +
                                       y * input_width + x +
                                       ky * input_width + kx;

                                t[j_out][i_in] = co * c_in * kh * kw +
                                                 ci * kh * kw +
                                                 ky * kw +
                                                 kx;
                            }
                        }


                    }
                }
            }
        }


    }

    const vector<vector<float>> &get_weights() {

        /*
         * 2D畳み込み版重み行列
         */

        for (int j = 0; j < n_out; ++j) {
            for (int i = 0; i < n_in; ++i) {
                if (t[j][i] == ConvLayerConst::T_WEIGHT_DISABLED) {
                    weights[j][i] = 0.f;
                } else {
                    weights[j][i] = h[t[j][i]];
                }
            }
        }
        return weights;
    }

};

#endif //CONVNETCPP_CONVLAYER_H
