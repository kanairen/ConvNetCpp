//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_DATA_H
#define CONVNETCPP_DATA_H

#include <sstream>
#include <vector>

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
};

template<class X, class Y>
class ImageDataSet : public DataSet<X, Y> {
public:
    unsigned int width;
    unsigned int height;
};

#endif //CONVNETCPP_DATA_H
