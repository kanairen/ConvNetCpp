//
//  test.cpp
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/15.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#include "test.h"

void mnist() {
    string filedir = "/Users/kanairen/Projects/xcode/ConvNetCpp/ConvNetCpp/res/mnist";
    string f_x_train = filedir + "/xtrain";
    string f_x_test = filedir + "/xtest";
    string f_y_train = filedir + "/ytrain";
    string f_y_test = filedir + "/ytest";
    unsigned int n_batch = 100;
    MNIST *mnist = MNIST::newMNIST(f_x_train, f_x_test, f_y_train, f_y_test, n_batch);
    vector<vector<vector<float>*>*>* x_train = mnist->getXTrain();
    vector<vector<vector<float>*>*>* x_test = mnist->getXTest();
    vector<vector<int>*>* y_train = mnist->getYTrain();
    vector<vector<int>*>* y_test = mnist->getYTest();
    
    Model *model = new Model();
    model->addLayer(28*28, 10, new Sigmoid(), 0.1);
    model->addLayer(10, 20, new Sigmoid(), 0.1);
    model->addLayer(20, 10, new Sigmoid(), 0.1);
    
    float train_error;
    float test_error;
    float sum_train_error;
    float sum_test_error;
    
    for (int i = 0; i < 100; i++) {
        sum_train_error = 0;
        sum_test_error = 0;
        for (int j = 0; j < n_batch; j++){
            cout << i + 1 << "st learning / " << j + 1 << "st batch" << endl;
            
            vector<int> *train_pred = model->forwardWithBackward((*x_train)[j], (*y_train)[j]);
            train_error = model->error(train_pred, (*y_train)[j]);
            cout << "train error : " << train_error << endl;
            sum_train_error += train_error;
            cout << "train error average : " << sum_train_error / (j + 1) << endl;
            
            vector<int> *test_pred = model->forward((*x_test)[j]);
            test_error = model->error(test_pred, (*y_test)[j]);
            cout << "test error : " << test_error << endl;
            sum_test_error += test_error;
            cout << "test error average : " << sum_test_error / (j + 1) << endl;
            
            cout << endl;
        }
    }
    
}
