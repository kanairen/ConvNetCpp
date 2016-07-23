//
// Created by 金井廉 on 2016/07/23.
//

#ifndef CONVNETCPP_TESTGRIDLAYER_H
#define CONVNETCPP_TESTGRIDLAYER_H

#include "../src/GridLayer.h"

namespace grid_layer {

    void test_init() {
        unsigned int n_data = 3;
        unsigned int input_width = 3;
        unsigned int input_height = 3;
        unsigned int c_in = 1;
        unsigned int c_out = 2;
        unsigned int kw = 2;
        unsigned int kh = 2;
        unsigned int sx = 1;
        unsigned int sy = 1;
        unsigned int px = 0;
        unsigned int py = 0;
        float (*activation)(float) = iden;
        float (*grad_activation)(float) = g_iden;
        bool is_weight_init_enabled = false;

        GridLayer2d_ layer(n_data, input_width, input_height, c_in, c_out, kw,
                           kh, sx, sy, px, py, activation, grad_activation,
                           is_weight_init_enabled);
    }

    void test_filter_outsize() {
        unsigned int input_width = 3;
        unsigned int input_height = 3;
        unsigned int c_in = 1;
        unsigned int c_out = 2;
        unsigned int kw = 2;
        unsigned int kh = 2;
        unsigned int sx = 1;
        unsigned int sy = 1;
        unsigned int px = 0;
        unsigned int py = 0;

        assert(GridLayer2d_::filter_outsize(28, kw, sx, px, false) == 27);
        assert(GridLayer2d_::filter_outsize(28, kh, sy, py, false) == 27);
    }

}


#endif //CONVNETCPP_TESTGRIDLAYER_H
