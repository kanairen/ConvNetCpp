//
//  test.cpp
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/15.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#include "test.h"

void mnist(string filedir) {
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
    
    Model *model = Model::newModel();
    model->addLayer(28*28, 128, new ReLU(), 0.1);
    model->addLayer(128, 64, new ReLU(), 0.1);
    model->addLayer(64, 10, new Sigmoid(), 0.1);
    
    vector<int> *train_pred;
    vector<int> *test_pred;
    float train_err;
    float test_err;
    float sum_train_err;
    float sum_test_err;
    
    for (int i = 0; i < 100; i++) {
        
        sum_train_err = 0;
        sum_test_err = 0;
        
        for (int j = 0; j < n_batch; j++){
            
            cout << i + 1 << "st learning / " << j + 1 << "st batch" << endl;
            
            train_pred = model->forwardWithBackward((*x_train)[j], (*y_train)[j]);
            train_err = model->error(train_pred, (*y_train)[j]);
            sum_train_err += train_err;
            
            cout << "train error : " << train_err << endl;
            cout << "train error average : " << sum_train_err / (j + 1) << endl;
            
            test_pred = model->forward((*x_test)[j]);
            test_err = model->error(test_pred, (*y_test)[j]);
            sum_test_err += test_err;
            
            cout << "test error : " << test_err << endl;
            cout << "test error average : " << sum_test_err / (j + 1) << endl;
            
            cout << endl;
        }
    }
    
}
