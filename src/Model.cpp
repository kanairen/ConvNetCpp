//
// Created by kanairen on 2016/06/14.
//

#include "Model.h"

const vector<vector<float>> &Model::forward(vector<vector<float>> &inputs) {
    const vector<vector<float>> *output = &inputs;
    for (int i = 0; i < layers.size(); i++) {
        output = &(layers[i].forward(*output));
    }
    return *output;
}

void Model::backward(const vector<vector<float>> &inputs,
                     const vector<vector<float>> &last_delta,
                     float learning_rate) {

    layers.back().backward(last_delta, layers[layers.size() - 2].get_z(),
                           learning_rate);
    for (int i = layers.size() - 2; i > 0; --i) {
        layers[i].backward(layers[i + 1], layers[i - 1].get_z(), learning_rate);
    }
    layers[0].backward(layers[1], inputs, learning_rate);
}

void Model::softmax(const vector<vector<float>> &outputs,
                    vector<vector<float>> &y) {

    if (outputs.size() != y.size() && outputs[0].size() != y[0].size()) {
        std::cerr << "error :  Model::softmax()" << endl;
        exit(1);
    }

    float u, sum_exp, max_output;
    for (int i = 0; i < outputs.size(); ++i) {
        sum_exp = 0.f;
        max_output = (*std::max_element(outputs[i].begin(), outputs[i].end()));
        for (int j = 0; j < outputs[i].size(); ++j) {
            u = exp(outputs[i][j] - max_output);
            y[i][j] = u;
            sum_exp += u;
        }
        for (int k = 0; k < outputs[i].size(); ++k) {
            y[i][k] /= sum_exp;
        }
    }

}

void Model::argmax(const vector<vector<float>> &y, vector<int> &predict) {
    if (y[0].size() != predict.size()) {
        std::cerr << "error :  Model::argmax()" << endl;
        exit(1);
    }

    float max, tmp;
    int max_idx;
    for (int i = 0; i < y[0].size(); ++i) {
        max = y[0][i];
        max_idx = 0;
        for (int j = 1; j < y.size(); ++j) {
            tmp = y[j][i];
            if (tmp > max) {
                max = tmp;
                max_idx = j;
            }
        }
        predict[i] = max_idx;
    }

}

float Model::error(const vector<int> &predict, const vector<int> &answer) {

    if (predict.size() != answer.size()) {
        std::cerr << "error :  Model::error()" << endl;
        exit(1);
    }

    float num_error = 0.;
    for (int i = 0; i < predict.size(); ++i) {
        if (predict[i] != answer[i]) {
            num_error += 1;
        }
    }

    return num_error / predict.size();

}