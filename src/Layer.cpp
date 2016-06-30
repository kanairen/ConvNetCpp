//
// Created by kanairen on 2016/06/13.
//

#import "Layer.h"

Layer::Layer(unsigned int n_data, unsigned int n_in, unsigned int n_out,
             float (*activation)(float), float (*grad_activation)(float))
        : AbstractLayer(n_data, n_in, n_out, activation, grad_activation),
          weights(vector<vector<float>>(n_out, vector<float>(n_in, 0.f))) {

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

const vector<vector<float>> &Layer::forward(
        const vector<vector<float>> &input) {
    float out;
    for (int i_data = 0; i_data < n_data; ++i_data) {
        for (int i_out = 0; i_out < n_out; ++i_out) {
            out = 0.f;
            for (int i_in = 0; i_in < n_in; ++i_in) {
                out += weights[i_out][i_in] * input[i_in][i_data];
            }
            out += biases[i_out];
            u[i_out][i_data] = out;
            z[i_out][i_data] = activation(out);
        }
    }
    return z;
}

void Layer::backward(const vector<vector<float>> &last_delta,
                     const vector<vector<float>> &prev_output,
                     const float learning_rate) {
#ifdef DEBUG_LAYER
    if (last_delta.size() != delta.size() &&
        last_delta[0].size() != delta[0].size()) {
        std::cerr << "Layer::backward : size of delta is not correct.";
        exit(1);
    }
#endif

    // 出力層のデルタとしてコピー
    unsigned long last_delta_length = last_delta[0].size();
    for (int i = 0; i < last_delta.size(); ++i) {
        for (int j = 0; j < last_delta_length; ++j) {
            delta[i][j] = last_delta[i][j];
        }
    }

    // パラメタ更新
    update(prev_output, learning_rate);

}

void Layer::backward(const AbstractLayer &next,
                     const vector<vector<float>> &prev_output,
                     const float learning_rate) {
    const vector<vector<float>> &next_weights = next.get_weights();
    const vector<vector<float>> &next_delta = next.get_delta();
    const unsigned int next_n_out = next.get_n_out();
    float d;
    for (int i_data = 0; i_data < n_data; ++i_data) {
        for (int i_out = 0; i_out < n_out; ++i_out) {
            d = 0.f;
            for (int i_n_out = 0; i_n_out < next_n_out; ++i_n_out) {
                // デルタを導出
                d += next_weights[i_n_out][i_out] * next_delta[i_n_out][i_data];
            }
            delta[i_out][i_data] = d * grad_activation(u[i_out][i_data]);
        }
    }

    // パラメタ更新
    update(prev_output, learning_rate);

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
