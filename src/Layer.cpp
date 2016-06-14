//
// Created by 金井廉 on 2016/06/13.
//

#import "Layer.h"

const vector<vector<float>> &Layer::forward(vector<vector<float>> &input) {
    for (int i_data = 0; i_data < n_data; ++i_data) {
        for (int i_out = 0; i_out < n_out; ++i_out) {
            for (int i_in = 0; i_in < n_in; ++i_in) {
                out_forward[i_out][i_data] =
                        activation(weights[i_out][i_in] * input[i_in][i_data] +
                                   biases[i_out]);
            }
        }
    }
    return out_forward;
}

