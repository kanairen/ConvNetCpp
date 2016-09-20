//
// Created by 金井廉 on 2016/08/01.
//

#ifndef CONVNETCPP_TRIANGLEGRIDLAYER_H
#define CONVNETCPP_TRIANGLEGRIDLAYER_H

#include "Layer.h"

class TriangularLatticeLayer : public Layer_ {
protected:

    unsigned int n_channel;
    unsigned int kernel_size;
    unsigned int stride;
    unsigned int padding;

public:

    TriangleGridLayer(unsigned int n_data,
                      unsigned int n_in, unsigned int n_out,
                      unsigned int n_channel, unsigned int kernel_size,
                      unsigned int stride, unsigned int padding,
                      float (*activation)(float),
                      float (*grad_activation)(float))
            : Layer_(n_data, n_in, n_out, activation, grad_activation),
              n_channel(n_channel),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding) { }

    virtual ~TriangleGridLayer() { }

};

#endif //CONVNETCPP_TRIANGLEGRIDLAYER_H

