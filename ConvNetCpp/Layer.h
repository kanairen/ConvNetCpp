//
//  Layer.h
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/06.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#ifndef LAYER_H_
#define LAYER_H_

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <random>

#include "Activation.h"

using namespace std;

class Layer{
private:
    
    int n_in, n_out;
    float learning_rate;
    
    Activation *activation;
    
    vector<vector<float> > *weights;
    vector<float> *biases;
    
    /* 計算結果を格納する配列・変数*/
    vector<float> *u;
    vector<float> *z;
    vector<float> *delta;
    float b_delta;
   
    Layer();
    Layer(const Layer& otherLayer);
    Layer(int n_in, int n_out, Activation *activation, float learning_rate);
    Layer& operator=(const Layer& otherLayer);
    
    void update();
    
public:
    virtual ~Layer();
    
    static Layer* newLayer(int n_in, int n_out, Activation *activation, float learning_rate){return new Layer(n_in, n_out, activation, learning_rate);}
    
    int getNIn(){return n_in;}
    int getNOut(){return n_out;}
    vector<vector<float> >* getWeights(){return weights;}
    vector<float>* getBiases(){return biases;}
    vector<float>* getDelta(){return delta;}
    
    virtual vector<float>* forward(vector<float> *x);
    void backward(vector<float> *nextDelta, vector<vector<float> > *nextWeight);// 逆伝播関数
    
    virtual string toString();
};


#endif
