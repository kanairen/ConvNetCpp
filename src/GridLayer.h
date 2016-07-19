//
// Created by kanairen on 2016/07/04.
//

#ifndef CONVNETCPP_GRIDLAYER_H
#define CONVNETCPP_GRIDLAYER_H

#import "Layer.h"

class GridLayer2d : public Layer {

    /*
     * 入力を二次元画像とする、ニューラルネットワークの隠れ層クラス
     */

protected:

    unsigned int input_width;
    unsigned int input_height;

    unsigned int output_width;
    unsigned int output_height;

    unsigned int c_in;
    unsigned int c_out;

    unsigned int kw;
    unsigned int kh;

    unsigned int sx;
    unsigned int sy;

    unsigned int px;
    unsigned int py;

    const unsigned int filter_outsize(unsigned int size, unsigned int k,
                                      unsigned int s, unsigned int p,
                                      bool is_covored_all) {
        /*
         * 出力画像の一辺のサイズを導出する
         */

        if (is_covored_all) {
            return ((size + p * 2 - k + s - 1) / s) + 1;
        } else {
            return ((size + p * 2 - k) / s) + 1;
        }

    }

public:

    GridLayer2d(unsigned int n_data,
                unsigned int input_width, unsigned int input_height,
                unsigned int c_in, unsigned int c_out,
                unsigned int kw, unsigned int kh,
                unsigned int sx, unsigned int sy,
                unsigned int px, unsigned int py,
                float (*activation)(float), float (*grad_activation)(float),
                bool is_weight_init_enabled = false)
            : input_width(input_width), input_height(input_height),
              output_width(filter_outsize(input_width, kw, sx, 0, false)),
              output_height(filter_outsize(input_height, kh, sy, 0, false)),
              c_in(c_in), c_out(c_out), kw(kw), kh(kh), sx(sx), sy(sy), px(px),
              py(py),
              Layer(n_data, c_in * input_width * input_height,
                    c_out * filter_outsize(input_height, kh, sy, py, false) *
                    filter_outsize(input_width, kw, sx, px, false),
                    activation, grad_activation, is_weight_init_enabled) { }

    virtual unsigned int get_output_width() { return output_width; }

    virtual unsigned int get_output_height() { return output_height; }

    const vector<float> &forward(const vector<float> &input) {

        /*
         * 入力の重み付き和を順伝播する関数
         *
         * input : n_in行 n_data列 の入力データ
         */

#ifdef PROFILE_ENABLED
        time_t start = clock();
#endif

        const vector<float> &w = get_weights();

        float out;
        const int n_d = n_data;
        const int n_o = n_out;
        const int n_i = n_in;
        int i_data, i_out, i_in, idx_output;

        float w_elem;

        std::fill(u.begin(), u.end(), 0.f);
        std::fill(z.begin(), z.end(), 0.f);

        for (i_out = 0; i_out < n_o; ++i_out) {
            for (i_in = 0; i_in < n_i; ++i_in) {
                w_elem = w[i_out * n_i + i_in];
                if (w_elem != 0) {
                    for (i_data = 0; i_data < n_d; ++i_data) {
                        out = biases[i_out] +
                              w_elem * input[i_in * n_d + i_data];
                        idx_output = i_out * n_d + i_data;
                        u[idx_output] += out;
                        z[idx_output] += activation(out);
                    }
                }
            }
        }

#ifdef PROFILE_ENABLED
        std::cout << "Layer::forward : " <<
        (float) (clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif

        return z;
    }
};


#endif //CONVNETCPP_GRIDLAYER_H
