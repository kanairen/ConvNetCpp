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
    vector<vector<float>*> *x_train;
    vector<vector<float>*> *x_test;
    vector<int> *y_train;
    vector<int> *y_test;
    
    static int toInteger(int i);
    
    vector<vector<float>*>* loadData(string filename);
    vector<int>* loadLabels(string filename);
    
public:
    MNIST(string fname_x_train, string fname_x_test, string fname_y_train, string fname_y_test);
    ~MNIST();
    
    vector<vector<float>*>* getXTrain(){return x_train;}
    vector<vector<float>*>* getXTest(){return x_test;}
    vector<int>* getYTrain(){return y_train;}
    vector<int>* getYTest(){return y_test;}
   
};

#endif /* MNIST_H_ */
