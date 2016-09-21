//
// Created by kanairen on 2016/07/04.
//

#ifndef CONVNETCPP_GRIDLAYER_H
#define CONVNETCPP_GRIDLAYER_H

#import "Layer.h"


class GridLayer2d_ : public Layer_ {

/*
 * 入力を二次元画像とする、ニューラルネットワークの隠れ層クラス
 */

//TODO cover all of output_size

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

public:

    static const unsigned int filter_outsize(unsigned int size, unsigned int k,
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

    GridLayer2d_(unsigned int n_data,
                 unsigned int input_width, unsigned int input_height,
                 unsigned int c_in, unsigned int c_out,
                 unsigned int kw, unsigned int kh,
                 unsigned int sx, unsigned int sy,
                 unsigned int px, unsigned int py,
                 float (*activation)(float), float (*grad_activation)(float),
                 bool is_weight_rand_init_enabled = false,
                 bool is_dropout_enabled = false,
                 float dropout_rate = 0.5f)
            : input_width(input_width), input_height(input_height),
              output_width(filter_outsize(input_width, kw, sx, 0, false)),
              output_height(filter_outsize(input_height, kh, sy, 0, false)),
              c_in(c_in), c_out(c_out), kw(kw), kh(kh), sx(sx), sy(sy), px(px),
              py(py),
              Layer_(n_data, c_in * input_width * input_height,
                     c_out * filter_outsize(input_height, kh, sy, py, false) *
                     filter_outsize(input_width, kw, sx, px, false),
                     activation, grad_activation, is_weight_rand_init_enabled,
                     is_dropout_enabled, dropout_rate) { }

    virtual unsigned int get_output_width() { return output_width; }

    virtual unsigned int get_output_height() { return output_height; }

};

#endif //CONVNETCPP_GRIDLAYER_H
