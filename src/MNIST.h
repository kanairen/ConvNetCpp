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

class MNIST : public DataSet<float, int> {
private:

    static int toInteger(int i);

    void loadData(std::string f_name, vector<vector<float>> &dst);

    void loadLabels(string f_name, vector<int> &dst);

public:

    MNIST(string f_x_train, string f_x_test, string f_y_train,
          string f_y_test) {
        loadData(f_x_train, x_train);
        loadData(f_x_test, x_test);
        loadLabels(f_y_train, y_train);
        loadLabels(f_y_test, y_test);
    };

    ~MNIST() { };

};


#endif //CONVNETCPP_MNIST_H
