//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_OPTIMIZER_H
#define CONVNETCPP_OPTIMIZER_H

#include <vector>

#include "Data.h"
#include "Layer.h"
#include "Model.h"

using std::vector;

template<class X, class Y>
void optimize(DataSet<X, Y> &data,
              vector<Layer> &layers,
              float learning_rate,
              unsigned int batch_size,
              unsigned int n_iter,
              unsigned int n_class) {

    /*
     * Modelオブジェクトを最適化する関数
     *
     * data : 学習・テスト用データセット
     * learning_rate : 学習率
     * input_size : １データあたりのベクトルサイズ
     * batch_size : １バッチデータあたりのサイズ
     * n_iter : 学習繰り返し回数/データセットを何周するか決めます
     * n_class : クラス数
     *
     */

    // 訓練データの分割数
    unsigned long n_batch_train = data.x_train[0].size() / batch_size;
    // テストデータの分割数
    unsigned long n_batch_test = data.x_test[0].size() / batch_size;

    cout << "n_batch_train : " << n_batch_train << endl;
    cout << "n_batch_test : " << n_batch_test << endl;

    // 分割データセット配列
    vector<vector<vector<X>>> x_trains(n_batch_train,
                                       vector<vector<X>>(data.xv_size(),
                                                          vector<X>(
                                                                  batch_size)));
    vector<vector<vector<X>>> x_tests(n_batch_test,
                                      vector<vector<X>>(data.xv_size(),
                                                         vector<X>(
                                                                 batch_size)));
    vector<vector<Y>> y_trains(n_batch_train, vector<Y>(batch_size));
    vector<vector<Y>> y_tests(n_batch_test, vector<Y>(batch_size));

    // 元のデータセットを分割データセット配列にコピー
    for (int i = 0; i < n_batch_train; ++i) {

        for (int k = 0; k < batch_size; ++k) {
            for (int j = 0; j < data.xv_size(); ++j) {
                x_trains[i][j][k] = data.x_train[j][i * batch_size + k];
            }
            y_trains[i][k] = data.y_train[i * batch_size + k];
        }

        if (i % (n_batch_train / n_batch_test) == 0) {
            unsigned long idx = i / (n_batch_train / n_batch_test);
            for (int k = 0; k < batch_size; ++k) {
                for (int j = 0; j < data.xv_size(); ++j) {
                    x_tests[idx][j][k] = data.x_test[j][idx * batch_size + k];
                }
                y_tests[idx][k] = data.y_test[idx * batch_size + k];
            }
        }

    }

    // 学習モデルオブジェクト
    Model model(layers, batch_size);

    // softmaxの結果を格納する配列
    vector<vector<X>> sm_train(n_class, vector<X>(batch_size));
    vector<vector<X>> sm_test(n_class, vector<X>(batch_size));

    // argmaxの結果を格納する配列
    vector<Y> pred_train(batch_size);
    vector<Y> pred_test(batch_size);

    std::cout << "\nlearning start !\n" << std::endl;

    float batch_error_train, batch_error_test, average_error_train, average_error_test;

    for (int i = 0; i < n_iter; ++i) {

        average_error_train = 0.f;
        average_error_test = 0.f;

        std::cout << i + 1 << "th iteration\n";

        // データセット一周当たりの学習・テスト時間を計測
        clock_t start = clock();

        for (int j = 0; j < n_batch_train; ++j) {

            const vector<vector<X>> &train_output = model.forward(x_trains[j]);

            model.softmax(train_output, sm_train);
            model.argmax(sm_train, pred_train);

            batch_error_train = model.error(pred_train, y_trains[j]);
            average_error_train += batch_error_train;

            // 出力層デルタ
            for (int k = 0; k < batch_size; ++k) {
                sm_train[y_trains[j][k]][k] -= 1.f;
            }

            // 逆伝播
            model.backward(x_trains[j], sm_train, learning_rate);

            // テスト
            if (j % (n_batch_train / n_batch_test) == 0) {

                unsigned long idx = j / (n_batch_train / n_batch_test);

                const vector<vector<X>> &test_output = model.forward(
                        x_tests[idx]);

                model.softmax(test_output, sm_test);
                model.argmax(sm_test, pred_test);

                batch_error_test = model.error(pred_test, y_tests[idx]);
                average_error_test += batch_error_test;

            }

        }

        cout << "average error(training data):" <<
        average_error_train / n_batch_train << "\n";

        cout << "average error(test data):" <<
        average_error_test / n_batch_test << "\n";

        cout << "process time:" << (float) (clock() - start) / 1000000 << "s\n";

        cout << endl;

    }

}


#endif //CONVNETCPP_OPTIMIZER_H
