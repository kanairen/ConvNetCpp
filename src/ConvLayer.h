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
    vector<int> t;

    void update(const vector<float> &prev_output, const float learning_rate) {

        /*
         * フィルタ重み・バイアス値を更新する
         *
         * prev_output : 前層の出力
         * learning_rate : 学習率(0≦learning_rate≦1)
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        float dw, db, d;

        const int n_o = n_out;
        const int n_i = n_in;
        const int n_d = n_data;
        int i_out, i_in, i_data;

        for (i_out = 0; i_out < n_o; ++i_out) {
            for (i_in = 0; i_in < n_i; ++i_in) {
                if (t[i_out * n_i + i_in] !=
                    ConvLayerConst::T_WEIGHT_DISABLED) {
                    dw = 0.f;
                    db = 0.f;
                    for (i_data = 0; i_data < n_d; ++i_data) {
                        d = delta[i_out * n_d + i_data];
                        dw += d * prev_output[i_in * n_d + i_data];
                        db += d;
                    }
                    // n_dataで割る必要はない？
                    h[t[i_out * n_i + i_in]] -= learning_rate * dw;
                }
            }
            biases[i_out] -= learning_rate * db / n_d;
        }

#ifdef PROFILE_ENABLED
        std::cout << "ConvLayer2d::update : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

    }

public:

    ConvLayer2d(unsigned int n_data,
                unsigned int input_width, unsigned int input_height,
                unsigned int c_in, unsigned int c_out,
                unsigned int kw, unsigned int kh,
                unsigned int sx, unsigned int sy,
                unsigned int px, unsigned int py,
                float (*activation)(float), float (*grad_activation)(float))
            : GridLayer2d(n_data, input_width, input_height, c_in, c_out, kw,
                          kh, sx, sy, px, py, activation, grad_activation,
                          false),
              h(vector<float>(kw * kh * c_in * c_out)),
              t(vector<int>(n_out * n_in, ConvLayerConst::T_WEIGHT_DISABLED)) {

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
        for (int co = 0; co < c_out; ++co) {
            for (int ci = 0; ci < c_in; ++ci) {
                for (int y = 0; y < input_height - kh; y += sy) {
                    for (int x = 0; x < input_width - kw; x += sx) {

                        j_out = co * output_height * output_width +
                                y / sy * output_width +
                                x / sx;

                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {
                                i_in = ci * input_width * input_height +
                                       y * input_width + x +
                                       ky * input_width + kx;

                                t[j_out * n_in + i_in] = co * c_in * kh * kw +
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

    const vector<float> &get_weights() {

        /*
         * 2D畳み込み版重み行列
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        for (int j = 0; j < n_out; ++j) {
            for (int i = 0; i < n_in; ++i) {
                if (t[j * n_in + i] == ConvLayerConst::T_WEIGHT_DISABLED) {
                    weights[j * n_in + i] = 0.f;
                } else {
                    weights[j * n_in + i] = h[t[j * n_in + i]];
                }
            }
        }

#ifdef PROFILE_ENABLED
        std::cout << "ConvLayer2d::get_weights : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

        return weights;
    }

};

#endif //CONVNETCPP_CONVLAYER_H
