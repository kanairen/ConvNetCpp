//
// Created by kanairen on 2016/06/14.
//

#ifndef CONVNETCPP_MNIST_H
#define CONVNETCPP_MNIST_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Data.h"

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::vector;
using std::string;

class MNIST : public ImageDataSet<float, int> {
private:

    static int toInteger(int i);

    void loadData(std::string f_name, vector<vector<float>> &dst,
                  unsigned int &dst_n_row, unsigned int &dst_n_col);

    void loadLabels(string f_name, vector<int> &dst);

public:

    MNIST(string f_x_train, string f_x_test, string f_y_train,
          string f_y_test) {
        loadData(f_x_train, x_train, width, height);
        loadData(f_x_test, x_test, width, height);
        loadLabels(f_y_train, y_train);
        loadLabels(f_y_test, y_test);
    };

    ~MNIST() { };

    virtual unsigned int data_size() {
        return width * height;
    }


};


#endif //CONVNETCPP_MNIST_H
