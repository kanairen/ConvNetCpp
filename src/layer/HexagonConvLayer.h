//
// Created by Ren Kanai on 2016/09/29.
//

#ifndef CONVNETCPP_HEXAGONCONVLAYER_H
#define CONVNETCPP_HEXAGONCONVLAYER_H

#include "ConvLayer.h"

class HexagonConvLayer2d : ConvLayer2d_ {

    /*
     *  正三角形から成る多面体より生成した形状マップを入力とする、
     *  六角形のグリッドを持つニューラルネットワークの畳み込み層クラス
     */

private:

    // 成分数 (フィルタ幅)^2 × (入力チャネル) × （出力チャネル） のフィルタベクトル
    VectorXf h;

    // 入力画素(i, j)と重なるフィルタのインデックスを保持するベクトル
    MatrixXi t;

    HexagonConvLayer2d(unsigned int n_data,
                       unsigned int input_width, unsigned int input_height,
                       unsigned int c_in, unsigned int c_out,
                       unsigned int k, unsigned int s,
                       unsigned int px, unsigned int py,
                       float (*activation)(float),
                       float (*grad_activation)(float),
                       bool is_filter_rand_init_enabled = true,
                       float filter_constant_value = 0.f,
                       bool is_dropout_enabled = false,
                       float dropout_rate = 0.5f)
            : ConvLayer2d_(n_data, input_width, input_height, c_in, c_out,
                           2 * k + 1, 2 * k + 1, s, s, px, py, activation,
                           grad_activation, is_filter_rand_init_enabled,
                           filter_constant_value, is_dropout_enabled,
                           dropout_rate) {

        // フィルタが六角形になるようにtの一部の要素を無効にする
        int i_in, j_out;
        for (int co = 0; co < c_out; ++co) {
            for (int y = 0; y <= input_height - kh; y += sy) {
                for (int x = 0; x <= input_width - kw; x += sx) {

                    j_out = co * output_height * output_width +
                            y / sy * output_width +
                            x / sx;

                    for (int ci = 0; ci < c_in; ++ci) {
                        for (int ky = 0; ky < kh; ++ky) {
                            for (int kx = 0; kx < kw; ++kx) {

                                if (kx >= (int) (kh / 2) - ky &&
                                    kx < (kh - ky) + (int)(kh / 2)) {
                                    continue;
                                }

                                i_in = ci * input_width * input_height +
                                       y * input_width + x +
                                       ky * input_width + kx;

                                t(j_out,
                                  i_in) = ConvLayerConst::T_WEIGHT_DISABLED;

                            }
                        }
                    }

                }
            }
        }

        update_weights();

    }


};

#endif //CONVNETCPP_HEXAGONCONVLAYER_H
