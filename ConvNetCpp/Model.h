//
//  Model.h
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/06.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include <algorithm>
#include <float.h>
#include "Layer.h"

using namespace std;

class Model{
private:
    vector<Layer*> *layers;
    vector<int> *preds;
    vector<float> *delta;

    Model(const Model& model);
    Model& operator=(const Model& model);
    
    void backward(vector<float>* delta);
public:
    Model();
    virtual ~Model();

    void addLayer(int n_in, int n_out, float learning_rate=0.001);
    vector<int>* forward(vector<vector<float>*>* inputs);
    vector<int>* forwardWithBackward(vector<vector<float>*> *inputs,vector<int> *answers);
    static float error(vector<int>* predicts, vector<int>* answers);
    static int argmax(vector<float>* output);
};

#endif
