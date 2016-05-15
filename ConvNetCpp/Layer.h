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
#include <sstream>
#include <random>

using namespace std;

class Layer{
private:
    Layer* prev;
    Layer* next;
    
    float n_in, n_out;
    float learning_rate;
    vector<vector<float>> *weights;
    vector<float> *biases;
    
    /* 計算結果を格納する配列・変数*/
    vector<float> *u;
    vector<float> *z;
    vector<float> *delta;
    float b_delta;
   
    Layer();
    Layer(const Layer& otherLayer);
    Layer& operator=(const Layer& otherLayer);
    
    void init(int n_in, int n_out, float learning_rate);// メンバ初期化関数
    
    void backward();
    void update();
    
public:
    Layer(int n_in, int n_out, float learning_rate = 0.001);
    Layer(Layer* prev, int n_out, float learning_rate = 0.001);
    virtual ~Layer();
    
    vector<vector<float>>* getWeights(){return weights;}
    vector<float>* getBiases(){return biases;}
    vector<float>* getDelta(){return delta;}
    
    vector<float>* forward(vector<float> *x);
    void backward(vector<float>* delta);// 外部呼び出し用逆伝播関数
    
    virtual float activation(float x){return x;}
    virtual float gradActivation(float x){return 1;}
    
    virtual string toString();
};

#endif
