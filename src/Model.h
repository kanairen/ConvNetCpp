//
// Created by kanairen on 2016/06/14.
//

#ifndef CONVNETCPP_MODEL_H
#define CONVNETCPP_MODEL_H

#include <cmath>
#include <vector>
#include "layer/Layer.h"

using Eigen::VectorXi;
using std::vector;
using std::unique_ptr;


class Model_ {
private:
    vector<unique_ptr<Layer_>> &layers;

    Model_() = delete;

public:
    Model_(vector<unique_ptr<Layer_>> &layers, unsigned int n_data) :
            layers(layers) { }

    ~Model_() { }

    const vector<unique_ptr<Layer_> > &get_layers() { return layers; }

    const MatrixXf &forward(const MatrixXf &inputs, bool is_train) {

        /*
         * 全レイヤの順伝播
         *
         * inputs : 入力データ行列
         */

        const MatrixXf *output = &inputs;
        for (unique_ptr<Layer_> &layer : layers) {
            output = &(layer->forward(*output, is_train));
        }
        return *output;
    }

    void backward(const MatrixXf &inputs,
                  const MatrixXf &last_delta, float learning_rate) {

        /*
         * 全レイヤの逆伝播＋学習パラメタ更新
         *
         * inputs : 入力データ行列
         * last_delta : 出力層デルタ行列
         * learning_rate : 学習率 (0≦learning_rate≦1)
         */

        const MatrixXf *prev_output;
        const MatrixXf *next_weight = nullptr;
        const MatrixXf *next_delta = &last_delta;
        unsigned int next_n_out = layers.back()->get_n_out();

        for (int i = layers.size() - 1; i >= 0; --i) {

            prev_output = (i == 0) ? &inputs : &layers[i - 1]->get_z();

            layers[i]->backward(*next_weight,
                                *next_delta,
                                *prev_output,
                                next_n_out,
                                learning_rate);

            next_weight = &layers[i]->get_weights();

            next_delta = &layers[i]->get_delta();

            next_n_out = layers[i]->get_n_out();

        }

    }

    static void argmax(const MatrixXf &y, VectorXi &predict) {


        /*
         * 引数にとったベクトルy[i]中の最大値インデックスをpredict[i]に格納
         *
         * y : 入力ベクトル
         * predict : yの各列ベクトルの最大値インデックスを格納する配列
         */

        float tmp, max;
        int max_idx;
        for (int j = 0; j < y.cols(); ++j) {
            max = y(0, j);
            max_idx = 0;
            for (int i = 1; i < y.rows(); ++i) {
                tmp = y(i, j);
                if (tmp > max) {
                    max = tmp;
                    max_idx = i;
                }
            }
            predict(j) = max_idx;
        }

    }

    static float error(const VectorXi &predict, const VectorXi &answer) {
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

    static float error(const VectorXi &predict, const VectorXi &answer,
                       vector<int> &error_indices, vector<int> &error_answers,
                       int start_index) {
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
                error_indices.push_back(start_index + i);
                error_answers.push_back(answer[i]);
            }
        }

        return num_error / predict.size();

    }

};


#endif //CONVNETCPP_MODEL_H
