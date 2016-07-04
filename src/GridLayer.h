//
// Created by 金井廉 on 2016/07/04.
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

    unsigned int c_in;
    unsigned int c_out;

    unsigned int kw;
    unsigned int kh;

    unsigned int stride;

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
                unsigned int kw, unsigned int kh, unsigned int stride,
                float (*activation)(float), float (*grad_activation)(float))
            : Layer(n_data, c_in * input_width * input_height,
                    c_out * filter_outsize(input_width, kw, stride, 0, false) *
                    filter_outsize(input_height, kh, stride, 0, false),
                    activation, grad_activation),
              input_width(input_width), input_height(input_height),
              c_in(c_in), c_out(c_out), kw(kw), kh(kh), stride(stride) { }

};


#endif //CONVNETCPP_GRIDLAYER_H
