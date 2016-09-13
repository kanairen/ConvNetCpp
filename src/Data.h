//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_DATA_H
#define CONVNETCPP_DATA_H

#include <sstream>
#include <vector>
#include <algorithm>
#include <random>

using std::vector;

template<class X, class Y>
class DataSet {
public:
    vector<vector<X>> x_train;
    vector<vector<X>> x_test;
    vector<Y> y_train;
    vector<Y> y_test;

    virtual unsigned int data_size() = 0;

    std::string toString() {
        std::stringstream ss;
        ss << "x_train : " << x_train.size();
        ss << " x_test : " << x_test.size();
        ss << " y_train : " << y_train.size();
        ss << " y_test : " << y_test.size();
        return ss.str();
    }

    void shuffle(bool isTrain) {

        vector<vector<X>> &x = (isTrain) ? x_train : x_test;
        vector<Y> &y = (isTrain) ? y_train : y_test;

        vector<int> indices(x.size());

        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        std::shuffle(indices.begin(), indices.end(), std::mt19937());

        vector<vector<X>> tmp_x(indices.size());
        vector<Y> tmp_y(indices.size());

        for (int i = 0; i < indices.size(); ++i) {
            tmp_x[i] = x[indices[i]];
            tmp_y[i] = y[indices[i]];
        }

        for (int i = 0; i < indices.size(); ++i) {
            x[i] = tmp_x[i];
            y[i] = tmp_y[i];
        }

    }

    void cross_validation(unsigned int n_fold,
                          vector<vector<unsigned int>> &train_perm,
                          vector<vector<unsigned int>> &test_perm) {
        /*
         * n-fold交差検定としてのデータセットのインデックスパターン返す
         */

        // n-foldパターンの組み合わせを生成
        train_perm.resize(n_fold);
        test_perm.resize(n_fold);

        for (int i = 0; i < n_fold; ++i) {

            train_perm[i].resize(x_train.size());
            test_perm[i].resize(x_test.size());

            // 組み合わせの数（階乗）
            unsigned int n_pattern_train = 0;
            unsigned int n_pattern_test = 0;

            // 初期化
            for (int j = 0; j < x_train.size(); ++j) {
                train_perm[i][j] = j;
                n_pattern_train += j;
            }
            for (int j = 0; j < x_test.size(); ++j) {
                test_perm[i][j] = j;
                n_pattern_test += j;
            }

            // どの組み合わせを選択するか
            long pivot_train = rand() % n_pattern_train;
            long pivot_test = rand() % n_pattern_test;

            // 目的のパターンに行き着くまでループ
            for (int j = 0; j < pivot_train; j++) {
                std::next_permutation(train_perm[i].begin(),
                                      train_perm[i].end());
            }
            for (int j = 0; j < pivot_test; j++) {
                std::next_permutation(test_perm[i].begin(),
                                      test_perm[i].end());
            }

        }

    }


};

template<class X, class Y>
class ImageDataSet : public DataSet<X, Y> {
public:
    unsigned int width;
    unsigned int height;
};

#endif //CONVNETCPP_DATA_H
