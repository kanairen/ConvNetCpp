//
//  main.cpp
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/07.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#include <iostream>
#include "Model.h"

using namespace std;

void test_backward(){
    vector<vector<float>*> x_train;
    x_train.push_back(new vector<float> {1,2,3});
    x_train.push_back(new vector<float> {2,3,4});
    x_train.push_back(new vector<float> {1,2,3});
    x_train.push_back(new vector<float> {1,2,3});
    x_train.push_back(new vector<float> {2,3,4});
    vector<int> y{0,1,0,0,1};
    
    vector<vector<float>*> x_test;
    x_test.push_back(new vector<float> {1,2,3});
    x_test.push_back(new vector<float> {2,3,4});
    x_test.push_back(new vector<float> {2,3,4});
    
    
    Model *model = new Model();
    model->addLayer(3, 100);
    model->addLayer(100, 2);
    
    for(int k = 0; k < 1000; k++){
        cout << k << "st learning..." << endl;
        vector<int> *output = model->forwardWithBackward(&x_train, &y);
        for (int i=0; i < output->size(); i++) {
            cout << (*output)[i] << endl;
        }     
    }
    cout << "test : " << endl;
    vector<int> *test_output = model->forward(&x_test);
     for (int i=0; i < test_output->size(); i++) {
        cout << (*test_output)[i] << endl;
     }
}


int main(int argc, const char * argv[]) {
    test_backward();
    return 0;
}
