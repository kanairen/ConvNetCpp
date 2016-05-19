//
//  Model.cpp
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/06.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#include "Model.h"

Model::Model(){
    this->layers = new vector<Layer*>();
    this->preds = new vector<int>();
    this->delta = new vector<float>();
}

Model::~Model(){
    // レイヤ配列の解放
    for (auto itr = this->layers->begin(); itr != this->layers->end(); ++itr) {
        delete (*itr);
    }
    delete this->layers;
    delete this->preds;
    delete this->delta;
}

void Model::addLayer(int n_in, int n_out, Activation *activation, float learning_rate){
    this->layers->push_back(Layer::newLayer(n_in, n_out, activation, learning_rate));
}

vector<int>* Model::forward(vector<vector<float>*> *inputs){
    if (preds->size() != inputs->size()) {
        preds->resize(inputs->size());
    }
    for (int i = 0; i < inputs->size(); i++){
        vector<float>* x = (*inputs)[i];
        for(int j = 0; j < this->layers->size(); j++){
            Layer *layer = (*this->layers)[j];
            x = layer->forward(x);
        }
        (*this->preds)[i] = this->argmax(x);
    }
    return this->preds;
}

vector<int>* Model::forwardWithBackward(vector<vector<float>*> *inputs,vector<int> *answers){
    vector<int> *predicts = this->forward(inputs);
    if(predicts->size() != answers->size()){
        throw exception();
    }
    if(this->delta->size() != predicts->size()){
        this->delta->resize(predicts->size());
    }
    for (int i = 0; i < predicts->size(); i++) {
        (*this->delta)[i] = (*predicts)[i] - (*answers)[i];
    }
    this->backward(this->delta);
    return predicts;
}

void Model::backward(vector<float> *delta){
    (*this->layers)[this->layers->size() - 1]->backward(delta);
}

float Model::error(vector<int> *predicts, vector<int> *answers){
    if(predicts->size() != answers->size()){
        throw exception();
    }
    float err = 0.0;
    for(int i = 0; i < predicts->size(); i++){
        if((*predicts)[i] != (*answers)[i]){
            err += 1.;
        }
    }
    return err / predicts->size();
}

int Model::argmax(vector<float> *output){
    float max = FLT_MIN;
    int max_idx = 0;
    for(int i = 0; i < output->size(); i++){
        if((*output)[i] > max){
            max = (*output)[i];
            max_idx = i;
        }
    }
    return max_idx;
}
