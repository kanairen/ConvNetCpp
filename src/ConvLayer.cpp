//
// Created by kanairen on 2016/06/29.
//

#include "ConvLayer.h"

ConvLayer2d::ConvLayer2d(unsigned int n_data,
                         unsigned int input_width, unsigned int input_height,
                         unsigned int c_in, unsigned int c_out,
                         unsigned int kw, unsigned int kh, unsigned int stride,
                         float (*activation)(float),
                         float (*grad_activation)(float))
        : AbstractLayer(n_data, c_in * input_width * input_height,
                        ((input_width - kw) / stride) *
                        ((input_height - kh) / stride) * c_out,
                        activation, grad_activation),
          input_width(input_width), input_height(input_height),
          c_in(c_in), c_out(c_out), kw(kw), kh(kh), stride(stride),
          h(vector<float>(kw * kh * c_in * c_out)),
          t(vector<vector<int>>(n_out, vector<int>(n_in,
                                                   ConvLayerConst::T_WEIGHT_DISABLED))) {

    // 乱数生成器
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<float> uniform(-sqrtf(6.f / (n_in + n_out)),
                                                  sqrtf(6.f / (n_in + n_out)));

    // filter要素を初期化
    for (float &f : h) {
        f = uniform(mt);
    }

    // tを初期化
    int j_out, i_in;
    for (int co = 0; co < c_out; ++co) {
        for (int ci = 0; ci < c_in; ++ci) {
            for (int y = 0; y < input_height - kh; y += stride) {
                for (int x = 0; x < input_width - kw; x += stride) {

                    j_out = co * ((input_height - kh) / stride) *
                            ((input_width - kw) / stride) +
                            y / stride * ((input_width - kw) / stride) +
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

const vector<vector<float>> &ConvLayer2d::get_weights() {
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


void ConvLayer2d::update(const vector<vector<float>> &prev_output,
                         const float learning_rate) {
    float dw, db;

    for (int i_out = 0; i_out < n_out; ++i_out) {
        for (int i_in = 0; i_in < n_in; ++i_in) {
            dw = 0.f;
            db = 0.f;
            for (int i_data = 0; i_data < n_data; ++i_data) {
                // オーバフローを防ぐため、先に学習率を掛ける
                dw += learning_rate * delta[i_out][i_data] *
                      prev_output[i_in][i_data];
                db += learning_rate * delta[i_out][i_data];
            }
            if (t[i_out][i_in] != ConvLayerConst::T_WEIGHT_DISABLED) {
                h[t[i_out][i_in]] -= (dw / n_data);
            }
        }
        biases[i_out] -= (db / n_data);
    }

}
