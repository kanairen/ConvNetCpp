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
    if (input.size() != n_out) {

    }
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

    // 出力層のデルタとしてコピー
    std::copy(last_delta.begin(), last_delta.end(), delta.begin());

#ifdef SHOW_DELTA
    float d;
    float sum_delta = 0.f;
    float max_delta = -MAXFLOAT;
    float min_delta = MAXFLOAT;
    for (int i_out = 0; i_out < n_out; ++i_out) {
        for (int i_data = 0; i_data < n_data; ++i_data) {
            d = delta[i_out][i_data];
            sum_delta += d;
            if (d > max_delta) {
                max_delta = d;
            }
            if (d < min_delta) {
                min_delta = d;
            }
        }
    }
    cout << "sum_delta : " << sum_delta << endl;
    cout << "max_delta : " << max_delta << endl;
    cout << "min_delta : " << min_delta << endl;
#endif

    // パラメタ更新
    update(prev_output, learning_rate);

}

void Layer::backward(const Layer &next,
                     const vector<vector<float>> &prev_output,
                     const float learning_rate) {
    const vector<vector<float>> &next_weights = next.weights;
    const vector<vector<float>> &next_delta = next.delta;
    const unsigned int next_n_out = next.n_out;
    float d;
    for (int i_data = 0; i_data < n_data; ++i_data) {
        for (int i_out = 0; i_out < n_out; ++i_out) {
            d = 0.f;
            for (int i_n_out = 0; i_n_out < next_n_out; ++i_n_out) {
                // デルタを導出
                d += grad_activation(u[i_out][i_data]) *
                     next_weights[i_n_out][i_out] * next_delta[i_n_out][i_data];
            }
            delta[i_out][i_data] = d;
        }
    }

#ifdef SHOW_DELTA
    float sum_delta = 0.f;
    float max_delta = -MAXFLOAT;
    float min_delta = MAXFLOAT;
    for (int i_out = 0; i_out < n_out; ++i_out) {
        for (int i_data = 0; i_data < n_data; ++i_data) {
            d = delta[i_out][i_data];
            sum_delta += d;
            if (d > max_delta) {
                max_delta = d;
            }
            if (d < min_delta) {
                min_delta = d;
            }
        }
    }
    cout << "sum_delta : " << sum_delta << endl;
    cout << "max_delta : " << max_delta << endl;
    cout << "min_delta : " << min_delta << endl;
#endif

    // パラメタ更新
    update(prev_output, learning_rate);

}

void Layer::update(const vector<vector<float>> &prev_output,
                   const float learning_rate) {
    float dw, db;
#ifdef SHOW_DW
    float sum_dw = 0;
    float max_dw = -MAXFLOAT;
    float min_dw = MAXFLOAT;
#endif
    for (int i_out = 0; i_out < n_out; ++i_out) {
        for (int i_in = 0; i_in < n_in; ++i_in) {
            dw = 0.f;
            db = 0.f;
            for (int i_data = 0; i_data < n_data; ++i_data) {
                // オーバフローを防ぐため、先に学習率を掛ける
                dw += learning_rate * delta[i_out][i_data] *
                      prev_output[i_in][i_data];
                db += learning_rate * delta[i_out][i_data];
#ifdef SHOW_DW
                sum_dw += dw;
                if (dw > max_dw) {
                    max_dw = dw;
                }
                if (dw < min_dw) {
                    min_dw = dw;
                }
#endif
            }
            weights[i_out][i_in] -= dw / n_data;
        }
        biases[i_out] -= db / n_data;
    }

#ifdef SHOW_DW
    cout << "sum_dw : " << sum_dw << endl;
    cout << "max_dw : " << max_dw << endl;
    cout << "min_dw : " << min_dw << endl;
#endif

}
