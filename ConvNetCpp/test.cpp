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
    MNIST mnist(f_x_train, f_x_test, f_y_train, f_y_test);
    
    Model *model = new Model();
    model->addLayer(28*28, 128, 0.001);
    model->addLayer(128, 256, 0.001);
    model->addLayer(256, 10, 0.001);
    
    for (int i = 0; i < 1000; i++) {
        cout << i + 1 << "st learning..." << endl;
        vector<int> *train_pred = model->forwardWithBackward(mnist.getXTrain(), mnist.getYTrain());
        cout << "train error : " << model->error(train_pred, mnist.getYTrain()) << endl;
        vector<int> *test_pred = model->forward(mnist.getXTest());
        cout << "test error : " << model->error(test_pred, mnist.getYTest()) << endl;
    }
    
}
