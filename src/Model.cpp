//
// Created by kanairen on 2016/06/14.
//

#include "Model.h"

const vector<vector<float>> &Model::forward(
        const vector<vector<float>> &inputs) {

    /*
     * 全レイヤの順伝播
     *
     * inputs : n_in行 n_data列 の入力データ
     */

    const vector<vector<float>> *output = &inputs;
    for (Layer *layer : layers) {
        output = &(layer->forward(*output));
    }
    return *output;
}

void Model::backward(const vector<vector<float>> &inputs,
                     const vector<vector<float>> &last_delta,
                     float learning_rate) {

    /*
     * 全レイヤの逆伝播＋学習パラメタ更新
     *
     * inputs : n_in行 n_data列 の入力データ
     * last_delta : 出力層デルタ
     * learning_rate : 学習率(0≦learning_rate≦1)
     */

    const vector<vector<float>> *prev_output;
    for (int i = layers.size() - 1; i >= 0; --i) {

        prev_output = (i == 0) ? &inputs : &layers[i - 1]->get_z();

        if (i == layers.size() - 1) {
            layers[i]->backward(last_delta, *prev_output, learning_rate);
        } else {
            layers[i]->backward(layers[i + 1]->get_weights(),
                                layers[i + 1]->get_delta(), *prev_output,
                                learning_rate);
        }

    }

}

void Model::softmax(const vector<vector<float>> &outputs,
                    vector<vector<float>> &y) {

    /*
     * ソフトマックス関数
     *
     * output : 出力層の出力
     * y : softmax関数値を格納する配列
     */

#ifdef DEBUG_MODEL
    if (outputs.size() != y.size() && outputs[0].size() != y[0].size()) {
        std::cerr << "error :  Model::softmax()" << endl;
        exit(1);
    }
#endif

    float u, sum_exp, max_output;
    unsigned long outputs_size = outputs.size();
    for (int j = 0; j < outputs[0].size(); ++j) {

        // 最大出力値を求める
        max_output = FLT_MIN;
        for (int i = 0; i < outputs_size; ++i) {
            if (outputs[i][j] > max_output) {
                max_output = outputs[i][j];
            }
        }

        // softmax関数の分子・分母を求める
        sum_exp = 0.f;
        for (int i = 0; i < outputs_size; ++i) {
            u = expf(outputs[i][j] - max_output);
            y[i][j] = u;
            sum_exp += u;
        }

        // softmax関数の出力値を求める
        for (int i = 0; i < outputs_size; ++i) {
            y[i][j] /= sum_exp;
        }

    }
}

void Model::argmax(const vector<vector<float>> &y, vector<int> &predict) {

    /*
     * 引数にとったベクトルy[i]中の最大値インデックスをpredict[i]に格納
     *
     * y : 入力ベクトル
     * predict : yの各列ベクトルの最大値インデックスを格納する配列
     */

#ifdef DEBUG_MODEL
    if (y[0].size() != predict.size()) {
        std::cerr << "error :  Model::argmax()" << endl;
        exit(1);
    }
#endif

    float max, tmp;
    int max_idx;
    unsigned long y_size = y.size();
    for (int i = 0; i < y[0].size(); ++i) {
        max = y[0][i];
        max_idx = 0;
        for (int j = 1; j < y_size; ++j) {
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

    /*
     * predictとanswerの各要素を比較し、誤りの割合を返す
     *
     * predict : 正解ラベルの予測
     * answer : Ground-Truth
     */

#ifdef DEBUG_MODEL
    if (predict.size() != answer.size()) {
        std::cerr << "error :  Model::error()" << endl;
        exit(1);
    }
#endif

    float num_error = 0.f;
    for (int i = 0; i < predict.size(); ++i) {
        if (predict[i] != answer[i]) {
            num_error += 1;
        }
    }

    return num_error / predict.size();

}