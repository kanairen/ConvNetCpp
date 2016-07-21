//
// Created by kanairen on 2016/06/15.
//

#ifndef CONVNETCPP_TEST_MNIST_H
#define CONVNETCPP_TEST_MNIST_H

#include "../src/MNIST.h"
#include <unistd.h>
#include <iomanip>
#include <cassert>

void test_mnist() {
    MNIST mnist("./x_train", "./x_test", "./y_train", "./y_test");

    assert(mnist.x_train.size() == mnist.y_train.size());
    assert(mnist.x_test.size() == mnist.y_test.size());

    // train
    for (int i = 0; i < 5; ++i) {
        cout << mnist.y_train[i] << endl;
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                cout << std::setw(4) << mnist.x_train[i][row * 28 + col] << " ";
            }
            cout << endl;
        }
    }

    // test
    for (int i = 0; i < 5; ++i) {
        cout << mnist.y_test[i] << endl;
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                cout << std::setw(4) << mnist.x_test[i][row * 28 + col] << " ";
            }
            cout << endl;
        }
    }

}

#endif //CONVNETCPP_TEST_MNIST_H
