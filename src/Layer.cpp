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



