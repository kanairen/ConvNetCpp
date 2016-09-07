//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_OPTIMIZER_H
#define CONVNETCPP_OPTIMIZER_H

#include <vector>

#include "Data.h"
#include "Model.h"

using Eigen::VectorXi;
using std::vector;

template<class X, class Y>
void optimize(DataSet<X, Y> &data,
              vector<Layer *> &layers,
              float learning_rate,
              unsigned int batch_size,
              unsigned int n_iter,
              unsigned int n_class) {

    /*
     * Modelオブジェクトを最適化
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
    vector<vector<X>> x_trains(n_batch_train,
                               vector<X>(data.data_size() * batch_size));
    vector<vector<X>> x_tests(n_batch_test,
                              vector<X>(data.data_size() * batch_size));
    vector<vector<Y>> y_trains(n_batch_train, vector<Y>(batch_size));
    vector<vector<Y>> y_tests(n_batch_test, vector<Y>(batch_size));

    // 元のデータセットを分割データセット配列にコピー
    for (int i = 0; i < n_batch_train; ++i) {

        for (int k = 0; k < batch_size; ++k) {
            for (int j = 0; j < data.data_size(); ++j) {
                x_trains[i][j * batch_size + k] = data.x_train[j][
                        i * batch_size + k];
            }
            y_trains[i][k] = data.y_train[i * batch_size + k];
        }

        if (i % (n_batch_train / n_batch_test) == 0) {
            unsigned long idx = i / (n_batch_train / n_batch_test);
            for (int k = 0; k < batch_size; ++k) {
                for (int j = 0; j < data.data_size(); ++j) {
                    x_tests[idx][j * batch_size + k] = data.x_test[j][
                            idx * batch_size + k];
                }
                y_tests[idx][k] = data.y_test[idx * batch_size + k];
            }
        }

    }

    // 学習モデルオブジェクト
    Model model(layers, batch_size);

    // softmaxの結果を格納する配列
    vector<X> sm_train(n_class * batch_size);

    // argmaxの結果を格納する配列
    vector<Y> pred_train(batch_size);
    vector<Y> pred_test(batch_size);

    std::cout << "\nlearning start !\n" << std::endl;

    float batch_error_train, batch_error_test, average_error_train, average_error_test;

    clock_t start, train_start, test_start;

    for (int i = 0; i < n_iter; ++i) {

        average_error_train = 0.f;
        average_error_test = 0.f;

        // データセット一周当たりの学習・テスト時間を計測
        start = clock();

        for (int j = 0; j < n_batch_train; ++j) {

            std::cout << i + 1 << "th iteration / " << j + 1 << "th batch\n";

            train_start = clock();

            const vector<X> &train_output = model.forward(x_trains[j]);

            model.argmax(train_output, pred_train, n_class, batch_size);

            batch_error_train = model.error(pred_train, y_trains[j]);
            average_error_train += batch_error_train;

            cout << "batch error(training data):" << batch_error_train << "\n";
            cout << "average error(training data):" <<
            average_error_train / (j + 1) << "\n";

            // 出力層デルタ
            for (int k = 0; k < train_output.size(); ++k) {
                for (int l = 0; l < batch_size; ++l) {
                    sm_train[k * batch_size + l] = train_output[k * batch_size +
                                                                l];
                    if (k == y_trains[j][l]) {
                        sm_train[k * batch_size + l] -= 1.f;
                    }
                }
            }

            // 逆伝播
            model.backward(x_trains[j], sm_train, learning_rate);

            cout << "train time:" <<
            (float) (clock() - train_start) / CLOCKS_PER_SEC << "s\n";

            // テスト
            if (j % (n_batch_train / n_batch_test) == 0) {

                unsigned long idx = j / (n_batch_train / n_batch_test);

                test_start = clock();

                const vector<X> &test_output = model.forward(x_tests[idx]);

                model.argmax(test_output, pred_test, n_class, batch_size);

                batch_error_test = model.error(pred_test, y_tests[idx]);
                average_error_test += batch_error_test;

                cout << "batch error(test data):" << batch_error_test << "\n";
                cout << "average error(test data):" <<
                average_error_test / (idx + 1) << "\n";

                cout << "test time:" <<
                (float) (clock() - test_start) / CLOCKS_PER_SEC << "s\n";

            }

            cout << endl;

        }

        cout << "average error(training data):" <<
        average_error_train / n_batch_train << "\n";

        cout << "average error(test data):" <<
        average_error_test / n_batch_test << "\n";

        cout << "process time:" << (float) (clock() - start) / CLOCKS_PER_SEC <<
        "s\n";

        cout << endl;

    }

}


template<class X, class Y>
void optimize_(DataSet<X, Y> &data,
               vector<Layer_ *> &layers,
               float learning_rate,
               unsigned int batch_size,
               unsigned int n_iter,
               unsigned int n_class) {

    /*
     * Modelオブジェクトを最適化
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

    // 分割データセット行列
    vector<MatrixXf> x_trains(n_batch_train,
                              MatrixXf(data.data_size(), batch_size));
    vector<MatrixXf> x_tests(n_batch_test,
                             MatrixXf(data.data_size(), batch_size));
    vector<VectorXi> y_trains(n_batch_train, VectorXi(batch_size));
    vector<VectorXi> y_tests(n_batch_test, VectorXi(batch_size));

    // 元のデータセットを分割データセット配列にコピー
    for (int i = 0; i < n_batch_train; ++i) {
        for (int k = 0; k < batch_size; ++k) {
            for (int j = 0; j < data.data_size(); ++j) {
                x_trains[i](j, k) = data.x_train[j][i * batch_size + k];
            }
            y_trains[i](k) = data.y_train[i * batch_size + k];
        }
    }

    for (int i = 0; i < n_batch_test; ++i) {
        for (int k = 0; k < batch_size; ++k) {
            for (int j = 0; j < data.data_size(); ++j) {
                x_tests[i](j, k) = data.x_test[j][i * batch_size + k];
            }
            y_tests[i](k) = data.y_test[i * batch_size + k];
        }
    }

    // 学習モデルオブジェクト
    Model_ model(layers, batch_size);

    // softmaxの結果を格納する配列
    MatrixXf sm_train(n_class, batch_size);

    // argmaxの結果を格納する配列
    VectorXi pred_train(batch_size);
    VectorXi pred_test(batch_size);

    std::cout << "\nlearning start !\n" << std::endl;

    float batch_error_train, batch_error_test, average_error_train, average_error_test;

    clock_t start, train_start, test_start;

    for (int i = 0; i < n_iter; ++i) {

        average_error_train = 0.f;
        average_error_test = 0.f;

        // データセット一周当たりの学習・テスト時間を計測
        start = clock();

        for (int j = 0; j < n_batch_train; ++j) {

            std::cout << i + 1 << "th iteration / " << j + 1 << "th batch\n";

            train_start = clock();

            const MatrixXf &train_output = model.forward(x_trains[j]);

            model.argmax(train_output, pred_train);

            batch_error_train = model.error(pred_train, y_trains[j]);
            average_error_train += batch_error_train;

            cout << "batch error(training data):" << batch_error_train << "\n";
            cout << "average error(training data):" <<
            average_error_train / (j + 1) << "\n";

            // 出力層デルタ
            sm_train = train_output;
            for (int k = 0; k < batch_size; ++k) {
                sm_train(y_trains[j](k), k) -= 1.f;
            }

            // 逆伝播
            model.backward(x_trains[j], sm_train, learning_rate);

            cout << "train time:" <<
            (float) (clock() - train_start) / CLOCKS_PER_SEC << "s\n";

            // テスト
            if (j % (n_batch_train / n_batch_test) == 0) {

                unsigned long idx = j / (n_batch_train / n_batch_test);

                test_start = clock();

                const MatrixXf &test_output = model.forward(x_tests[idx]);

                model.argmax(test_output, pred_test);

                batch_error_test = model.error(pred_test, y_tests[idx]);
                average_error_test += batch_error_test;

                cout << "batch error(test data):" << batch_error_test << "\n";
                cout << "average error(test data):" <<
                average_error_test / (idx + 1) << "\n";

                cout << "test time:" <<
                (float) (clock() - test_start) / CLOCKS_PER_SEC << "s\n";

            }

            cout << endl;

        }

        cout << "average error(training data):" <<
        average_error_train / n_batch_train << "\n";

        cout << "average error(test data):" <<
        average_error_test / n_batch_test << "\n";

        cout << "process time:" << (float) (clock() - start) / CLOCKS_PER_SEC <<
        "s\n";

        cout << endl;

    }

}


#endif //CONVNETCPP_OPTIMIZER_H
