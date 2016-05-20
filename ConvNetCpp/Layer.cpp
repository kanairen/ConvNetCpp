//
//  Layer.cpp
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/06.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#include"Layer.h"


// メンバ初期化
Layer::Layer(unsigned int n_in, unsigned int n_out, Activation *activation, float learning_rate){
    
    this->n_in = n_in;
    this->n_out = n_out;
    this->activation = activation;
    this->learning_rate = learning_rate;
    
    // 乱数生成器
    float low = -sqrt(6./(n_in+n_out));
    float high = sqrt(6./(n_in+n_out));
    uniform_real_distribution<> real(low, high);
    mt19937 mt((random_device())());
    
    // パラメタ変数の初期化
    this->weights = new vector<vector<float> >(n_out);
    this->biases = new vector<float>(n_out);
    for(int i = 0; i < n_out; i++){
        (*this->biases)[i] = 0.0;
        (*this->weights)[i] = vector<float>(n_in);
        for(int j = 0; j < n_in; j++){
            (*this->weights)[i][j] = real(mt);
        }
    }
    
    this->u = new vector<float>(n_out);
    this->z = new vector<float>(n_out);
    this->delta = new vector<float>(n_out);
    this->b_delta = 0.0;
}

Layer::~Layer(){
    delete this->weights;
    delete this->biases;
    delete this->u;
    delete this->z;
    delete this->delta;
}

// 順伝播関数
// 伝播により、逆伝播に使う入力重み付き和uが求まる
vector<float>* Layer::forward(vector<float> *x){
    float u;
    for(int i = 0; i < this->n_out; i++){
        u = 0.0;
        for(int j = 0; j < this->n_in; j++){
            u += (*this->weights)[i][j] * (*x)[j];
        }
        u += (*this->biases)[i];
        (*this->u)[i] = u;
        (*this->z)[i] = this->activation->f(u);
    }
    return this->z;
}

// 逆伝播関数
void Layer::backward(vector<float> *nextDelta, vector<vector<float> > *nextWeight){
    if(nextWeight == NULL){
        if(this->delta->size() != nextDelta->size()){
            cerr << "error : backward on output layer." << endl;
            exit(1);
        }
        for (int i = 0; i < nextDelta->size(); i++) {
            (*this->delta)[i] = (*nextDelta)[i];
        }
        this->b_delta = 0.0;
        this->update();
        return;
    }
    
    // 重みパラメタの逆伝播（次層のバイアスパラメタは現在層の重みパラメタから影響を受けないので、）
    for(int j = 0; j < this->n_out; j++){
        (*this->delta)[j] = 0.0;
        for(int k = 0; k < nextDelta->size(); k++){
            (*this->delta)[j] += (*nextDelta)[k] * (*nextWeight)[k][j];
        }
        (*this->delta)[j] *= this->activation->gf((*this->u)[j]);
    }
    
    // バイアスパラメタへの逆伝播
    this->b_delta = 0.0;
    for (int k = 0; k < nextDelta->size(); k++) {
        this->b_delta += (*nextDelta)[k] * 1.0;
    }
    
    // パラメタ更新
    this->update(); 
}

void Layer::update(){
    for(int i = 0; i < this->n_out; i++){
        for(int j = 0; j < this->n_in; j++){
            (*this->weights)[i][j] -= (*this->delta)[i] * this->learning_rate;
        }
        (*this->biases)[i] -= this->b_delta * this->learning_rate;
    }
}

string Layer::toString(){
    stringstream ss;
    ss << "n_in : " << this->n_in << '\n';
    ss << "n_out : " << this->n_out << '\n';
    return ss.str();
}
