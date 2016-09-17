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

    virtual int data_size() = 0;

    static void shuffle(vector<vector<X>> &x, vector<Y> &y) {
        /*
         * 特徴ベクトル集合と正解データ集合を、特徴ベクトル・正解の関係が崩れないようシャッフルする
         */

        vector<int> indices(y.size());

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
//
//    static const vector<DataSet> *cross_validation(unsigned int n_fold,
//                                                   vector<vector<X>> x,
//                                                   vector<Y> y) {
//        /*
//         * n-fold交差検定のデータ・セットを返す
//         */
//
//        // shuffle
//        ShapeMapSet::shuffle(x, y);
//
//        unsigned int begin, end;
//        for (int i = 0; i < n_fold; ++i) {
//            begin = x.size() / n_fold * i;
//            end = x.size() / n_fold * (i + 1);
//
//        }
//    }

    std::string toString() {
        std::stringstream ss;
        ss << "x_train : " << x_train.size();
        ss << " x_test : " << x_test.size();
        ss << " y_train : " << y_train.size();
        ss << " y_test : " << y_test.size();
        return ss.str();
    }

};

template<class X, class Y>
class ImageDataSet : public DataSet<X, Y> {
public:
    int width;
    int height;
};

#endif //CONVNETCPP_DATA_H
