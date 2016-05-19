//
//  MNIST.h
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/14.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#ifndef MNIST_H_
#define MNIST_H_

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class MNIST{
private:
    vector<vector<vector<float>*>*> *x_train;
    vector<vector<vector<float>*>*> *x_test;
    vector<vector<int>*> *y_train;
    vector<vector<int>*> *y_test;
    
    MNIST();
    MNIST(string f_x_train, string f_x_test, string f_y_train, string f_y_test, unsigned int n_batch);
    
    static int toInteger(int i);
    
    vector<vector<vector<float>*>*>* loadData(string f_name, unsigned int n_batch);
    vector<vector<int>*>* loadLabels(string f_name, unsigned int n_batch);
    
public:
    virtual ~MNIST();
    static MNIST* newMNIST(string f_x_train, string f_x_test, string f_y_train, string f_y_test, unsigned int n_batch){return new MNIST(f_x_train, f_x_test, f_y_train, f_y_test, n_batch);}
    vector<vector<vector<float>*>*>* getXTrain(){return x_train;}
    vector<vector<vector<float>*>*>* getXTest(){return x_test;}
    vector<vector<int>*>* getYTrain(){return y_train;}
    vector<vector<int>*>* getYTest(){return y_test;}
   
};

#endif /* MNIST_H_ */
