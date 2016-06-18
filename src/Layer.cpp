//
// Created by kanairen on 2016/06/13.
//

#import "Layer.h"

Layer::Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
             float (*activation)(float), float (*grad_activation)(float)) :
        n_data(n_data), n_in(n_in), n_out(n_out), activation(activation),
        grad_activation(grad_activation),
        weights(vector<vector<float>>(n_out, vector<float>(n_in, 0.f))),
        biases(vector<float>(n_out, 0.f)),
        delta(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
        u(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))),
        z(vector<vector<float>>(n_out, vector<float>(n_data, 0.f))) {

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> uniform(-sqrt(6. / (n_in + n_out)),
                                             sqrt(6. / (n_in + n_out)));
    for (int i = 0; i < n_out; ++i) {
        for (int j = 0; j < n_in; ++j) {
            weights[i][j] = uniform(mt);
        }
    }

}

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

