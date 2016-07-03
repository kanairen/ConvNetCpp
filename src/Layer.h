//
// Created by kanairen on 2016/06/13.
//

#ifndef CONVNETCPP_LAYER_H
#define CONVNETCPP_LAYER_H

#include <random>
#include <vector>
#include <iostream>
#include "AbstractLayer.h"

using std::vector;
using std::cout;
using std::endl;

class Layer : public AbstractLayer {
private:

    void update(const vector<vector<float>> &prev_output,
                const float learning_rate);

public:
    Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
          float (*activation)(float), float (*grad_activation)(float));

    ~Layer() { }

};

#endif //CONVNETCPP_LAYER_H
