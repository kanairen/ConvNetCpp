//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_OPTIMIZER_H
#define CONVNETCPP_OPTIMIZER_H

#include <vector>

#include "Data.h"
#include "Model.h"
#include "util/IOUtil.h"

using Eigen::VectorXi;
using std::vector;

template<class X, class Y>
void optimize_(DataSet<X, Y> &data,
               vector<Layer_ *> &layers,
               float learning_rate,
               unsigned int batch_size,
               unsigned int n_iter,
               unsigned int n_class,
               string train_log_path,
               string test_log_path,
               float log_init_const_value = -1.f,
               bool is_error_example_enabled = false) {

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

    // ERRORの割合ログ
    std::vector<float> log_average_error_train(n_iter);
    std::vector<float> log_average_error_test(n_iter);
    std::fill(log_average_error_train.begin(), log_average_error_train.end(), log_init_const_value);
    std::fill(log_average_error_test.begin(), log_average_error_test.end(), log_init_const_value);

    std::cout << "\nlearning start !\n" << std::endl;

    float batch_error_train, batch_error_test,
            average_error_train, average_error_test;

    clock_t start, train_start, test_start;

    for (int i = 0; i < n_iter; ++i) {

        average_error_train = 0.f;
        average_error_test = 0.f;

        if (is_error_example_enabled) {

            // 誤りインデックス
            std::vector<int> error_indices_train;
            std::vector<int> error_indices_test;
            // 誤り推定
            std::vector<int> error_answers_train;
            std::vector<int> error_answers_test;

        }

        // データセット一周当たりの学習・テスト時間を計測
        start = clock();

        for (int j = 0; j < n_batch_train; ++j) {

            std::cout << i + 1 << "th iteration / " << j + 1 << "th batch\n";

            train_start = clock();

            // batchごとの誤りインデックス
            std::vector<int> error_indices_train_batch;
            std::vector<int> error_indices_test_batch;
            // batchごとの誤り推定
            std::vector<int> error_answers_train_batch;
            std::vector<int> error_answers_test_batch;

            const MatrixXf &train_output = model.forward(x_trains[j], true);

            model.argmax(train_output, pred_train);

            if (is_error_example_enabled) {
                batch_error_train = model.error(pred_train, y_trains[j],
                                                error_indices_train_batch,
                                                error_answers_train_batch,
                                                j * batch_size);
            } else {
                batch_error_train = model.error(pred_train, y_trains[j],
                                                j * batch_size);
            }

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

                const MatrixXf &test_output = model.forward(x_tests[idx],
                                                            false);

                model.argmax(test_output, pred_test);

                if (is_error_example_enabled) {
                    batch_error_test = model.error(pred_test, y_tests[idx],
                                                   error_indices_test_batch,
                                                   error_answers_test_batch,
                                                   idx * batch_size);
                } else {
                    batch_error_test = model.error(pred_test, y_tests[idx],
                                                   idx * batch_size);
                }

                average_error_test += batch_error_test;

                cout << "batch error(test data):" << batch_error_test << "\n";
                cout << "average error(test data):" <<
                     average_error_test / (idx + 1) << "\n";

                cout << "test time:" <<
                     (float) (clock() - test_start) / CLOCKS_PER_SEC << "s\n";

            }

            if (is_error_example_enabled) {
                std::copy(error_indices_train_batch.begin(),
                          error_indices_train_batch.end(),
                          std::back_inserter(error_indices_train));

                std::copy(error_indices_test_batch.begin(),
                          error_indices_test_batch.end(),
                          std::back_inserter(error_indices_test));

                std::copy(error_answers_train_batch.begin(),
                          error_answers_train_batch.end(),
                          std::back_inserter(error_answers_train));

                std::copy(error_answers_test_batch.begin(),
                          error_answers_test_batch.end(),
                          std::back_inserter(error_answers_test));
            }

            cout << endl;

        }

        average_error_train /= n_batch_train;
        cout << "average error(training data):" << average_error_train << "\n";
        log_average_error_train[i] = average_error_train;

        average_error_test /= n_batch_test;
        cout << "average error(test data):" << average_error_test << "\n";
        log_average_error_test[i] = average_error_test;

        cout << "incorrect estimations(training data) : \n";
        cout << "indices : ";
        print(error_indices_train);
        cout << "error estimations : ";
        print(error_answers_train);

        cout << "incorrect estimations(test data) : \n";
        cout << "indices : ";
        print(error_indices_test);
        cout << "error estimations : ";
        print(error_answers_test);

        cout << "process time:" << (float) (clock() - start) / CLOCKS_PER_SEC <<
             "s\n";

        cout << endl;

    }

    save_as_csv<float>(train_log_path, log_average_error_train);
    save_as_csv<float>(test_log_path, log_average_error_test);

}


#endif //CONVNETCPP_OPTIMIZER_H
