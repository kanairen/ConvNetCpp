//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_DATA_H
#define CONVNETCPP_DATA_H

#include <sstream>
#include <vector>
#include <set>
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

    unsigned int get_n_cls() {
        std::vector<Y> v;
        std::copy(y_train.begin(), y_train.end(), std::back_inserter(v));
        std::copy(y_test.begin(), y_test.end(), std::back_inserter(v));
        return std::set<Y>(v.begin(), v.end()).size();
    }

    static void shuffle(vector<vector<X>> &x, vector<Y> &y) {
        /*
         * 特徴ベクトル集合と正解データ集合を、特徴ベクトル・正解の関係が崩れないようシャッフルする
         */

        vector<int> indices(y.size());

        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        std::shuffle(indices.begin(), indices.end(), std::mt19937());

        vector<vector<X>> tmp_x(x.size(), vector<X>(x[0].size()));
        vector<Y> tmp_y(y.size());

        for (int i = 0; i < indices.size(); ++i) {
            for (int j = 0; j < x.size(); ++j) {
                tmp_x[j][i] = x[j][indices[i]];
            }
            tmp_y[i] = y[indices[i]];
        }

        for (int i = 0; i < indices.size(); ++i) {
            for (int j = 0; j < x.size(); ++j) {
                x[j][i] = tmp_x[j][i];
            }
            y[i] = tmp_y[i];
        }

    }

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
