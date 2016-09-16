//
// Created by kanairen on 2016/09/15.
//

#ifndef CONVNETCPP_DATASET_H
#define CONVNETCPP_DATASET_H

#include <sstream>
#include <vector>
#include <algorithm>
#include <random>

using std::vector;
using std::unique_ptr;

template<class X, class Y>
class BaseDataSet {

    /*
     * データセット基底クラス
     */

public:

    unique_ptr<vector<vector<X>>> x_train;
    unique_ptr<vector<vector<X>>> x_test;
    unique_ptr<vector<Y>> y_train;
    unique_ptr<vector<Y>> y_test;

protected:

    BaseDataSet(vector<vector<X>> *x_train, vector<vector<X>> *x_test,
                vector<Y> *y_train, vector<Y> *y_test)
            : x_train() { }

    static void shuffle(const unique_ptr<vector<vector<X>>> &x,
                        const unique_ptr<vector<Y>> &y) {
        /*
         * 特徴ベクトル集合と正解データ集合を、特徴ベクトル・正解の関係が崩れないようシャッフルする
         */

        vector<int> indices(x->size());

        for (int i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        std::shuffle(indices.begin(), indices.end(), std::mt19937());

        vector<vector<X>> tmp_x(indices.size());
        vector<Y> tmp_y(indices.size());

        for (int i = 0; i < indices.size(); ++i) {
            tmp_x[i] = (*x)[indices[i]];
            tmp_y[i] = (*y)[indices[i]];
        }

        for (int i = 0; i < indices.size(); ++i) {
            (*x)[i] = tmp_x[i];
            (*y)[i] = tmp_y[i];
        }

    }

public:
    void shuffle(bool is_train) {
        if (is_train) {
            BaseDataSet::shuffle(x_train, y_train);
        } else {
            BaseDataSet::shuffle(x_test, y_test);
        }
    }

    virtual unsigned int data_size() = 0;

    virtual static const BaseDataSet *load();

    static const vector<BaseDataSet> *cross_validation(unsigned int n_fold,
                                                       const vector<vector<X>> &x,
                                                       const vector<Y> &y) {
        /*
         * n-fold交差検定のデータ・セットを返す
         */

        vector<BaseDataSet> *data_sets = new vector<DataSet>();

        // shuffle
        ShapeMapSet::shuffle(x, y);

        unsigned int begin, end;
        for (int i = 0; i < n_fold; ++i) {
            begin = x.size() / n_fold * i;
            end = x.size() / n_fold * (i + 1);
            vector<vector<X>> *x_train = new vector<vector<X>>(
                    x.size() - (end - begin));
            vector<vector<X>> *x_test = new vector<vector<X>>(end - begin);
            vector<Y> *y_train = new vector<Y>(y.size() - (end - begin));
            vector<Y> *y_test = new vector<Y>(end - begin);

            std::copy(x.begin(), x.begin() + begin, x_train->begin());
            std::copy(y.begin(), y.begin() + begin, y_train->begin());

            std::copy(x.begin() + begin, x.begin() + end, x_test->begin());
            std::copy(y.begin() + begin, y.begin() + end, y_test->begin());

            std::copy(x.begin() + end, x.end(), std::back_inserter(x_train));
            std::copy(y.begin() + end, y.end(), std::back_inserter(y_train));

            data_sets->push_back(BaseDataSet(x_train, x_test, y_train, y_test));
        }

        return data_sets;
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
class ImageDataSet : public BaseDataSet<X, Y> {
public:
    unsigned int width;
    unsigned int height;
};


#endif //CONVNETCPP_DATASET_H
