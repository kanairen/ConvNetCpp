//
// Created by kanairen on 2016/06/15.
//

#include "config.h"
#include "MNIST.h"
#include "Layer.h"
#include "Model.h"

#ifdef CONV_NET_CPP_DEBUG

#include <iomanip>
#include "activation.h"

// コマンドライン引数にmnistへのパスを渡す
int main(int argc, char *argv[]) {

    // 繰り返し回数
    unsigned int n_iter = 1000;
    // 1バッチデータあたりのサイズ
    unsigned int batch_size = 10;
    // 入力サイズ
    unsigned int input_size = 28 * 28;
    // 出力クラス数
    unsigned int n_class = 10;

    // mnist
    MNIST mnist(argv[1], argv[2], argv[3], argv[4]);

    // 訓練データの分割数
    unsigned long n_batch_train = mnist.x_train[0].size() / batch_size;
    // テストデータの分割数
    unsigned long n_batch_test = mnist.x_test[0].size() / batch_size;

    cout << "n_batch_train : " << n_batch_train << endl;
    cout << "n_batch_test : " << n_batch_test << endl;

    // 分割データセット
    vector<vector<vector<float>>> x_trains(n_batch_train,
                                           vector<vector<float>>(input_size,
                                                                 vector<float>(
                                                                         batch_size)));
    vector<vector<vector<float>>> x_tests(n_batch_test,
                                           vector<vector<float>>(input_size,
                                                                 vector<float>(
                                                                         batch_size)));
    vector<vector<int>> y_trains(n_batch_train, vector<int>(batch_size));
    vector<vector<int>> y_tests(n_batch_test, vector<int>(batch_size));

    // MNISTデータセットを分割データセット配列にコピー
    for (int i = 0; i < n_batch_train; ++i) {

        for (int k = 0; k < batch_size; ++k) {
            for (int j = 0; j < input_size; ++j) {
                x_trains[i][j][k] = mnist.x_train[j][i * batch_size + k];
            }
            y_trains[i][k] = mnist.y_train[i * batch_size + k];
        }

        if (i % (n_batch_train / n_batch_test) == 0) {
            unsigned long idx = i / (n_batch_train / n_batch_test);
            for (int k = 0; k < batch_size; ++k) {
                for (int j = 0; j < input_size; ++j) {
                    x_tests[idx][j][k] = mnist.x_test[j][idx * batch_size + k];
                }
                y_tests[idx][k] = mnist.y_test[idx * batch_size + k];
            }
        }

    }

    // レイヤ配列
    vector<Layer> v{Layer(batch_size, input_size, n_class, iden, g_iden)};

    // 学習モデルオブジェクト
    Model model(v, batch_size);

    // 学習率
    float learning_rate = 0.001f;

    // softmaxの結果を格納する配列
    vector<vector<float>> sm_train(n_class, vector<float>(batch_size));
    vector<vector<float>> sm_test(n_class, vector<float>(batch_size));

    // argmaxの結果を格納する配列
    vector<int> pred_train(batch_size);
    vector<int> pred_test(batch_size);

    cout << "\nlearning start !\n" << endl;

    float batch_error_train, batch_error_test, average_error_train, average_error_test;

    for (int i = 0; i < n_iter; ++i) {

        average_error_train = 0.f;
        average_error_test = 0.f;

        cout << i + 1 << "th iteration\n";

        // データセット一周当たりの学習・テスト時間を計測
        clock_t start = clock();

        for (int j = 0; j < n_batch_train; ++j) {

            const vector<vector<float>> &train_output = model.forward(
                    x_trains[j]);

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

                const vector<vector<float>> &test_output = model.forward(
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

#endif