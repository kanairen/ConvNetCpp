//
// Created by kanairen on 2016/06/14.
//

#include "Model.h"

const vector<vector<float>> &Model::forward(
        const vector<vector<float>> &inputs) {

    /*
     * 全レイヤの順伝播
     *
     * inputs : 入力データ行列
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
     * inputs : 入力データ行列
     * last_delta : 出力層デルタ行列
     * learning_rate : 学習率 (0≦learning_rate≦1)
     */

//#ifdef DEBUG_MODEL
//    if (typeid(layers[0]) != typeid(SoftMaxLayer)) {
//        std::cerr <<
//        "ERROR(Model::backward()) : output layer is not SoftMaxLayer" <<
//        std::endl;
//        exit(1);
//    }
//#endif

    const vector<vector<float>> *prev_output;
    const vector<vector<float>> *next_weight = nullptr;
    const vector<vector<float>> *next_delta = &last_delta;

    for (int i = layers.size() - 1; i >= 0; --i) {

        prev_output = (i == 0) ? &inputs : &layers[i - 1]->get_z();

        layers[i]->backward(*next_weight,
                            *next_delta,
                            *prev_output,
                            learning_rate);

        next_weight = &layers[i]->get_weights();

        next_delta = &layers[i]->get_delta();

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