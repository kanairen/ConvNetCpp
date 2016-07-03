//
// Created by kanairen on 2016/06/13.
//

#import "Layer.h"

Layer::Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
             float (*activation)(float), float (*grad_activation)(float))
        : AbstractLayer(n_data, n_in, n_out, activation, grad_activation) {

    // 乱数生成器
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<float> uniform(-sqrtf(6.f / (n_in + n_out)),
                                                  sqrtf(6.f / (n_in + n_out)));
    // 重みパラメタの初期化
    for (vector<float> &row : weights) {
        for (float &w : row) {
            w = uniform(mt);
        }
    }

}



void Layer::update(const vector<vector<float>> &prev_output,
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
            weights[i_out][i_in] -= (dw / n_data);
        }
        biases[i_out] -= (db / n_data);
    }

}
