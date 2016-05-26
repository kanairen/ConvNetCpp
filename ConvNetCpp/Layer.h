//
//  Layer.h
//  ConvNetCpp
//
//  Created by ren on 2016/05/06.
//  Copyright © 2016年 ren. All rights reserved.
//

#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <sstream>
#include <random>
#include <Eigen/Core>

#include "Activation.h"

using namespace std;
using namespace Eigen;

class Layer {
private:

    unsigned int n_in, n_out, n_data;
    float learning_rate;

    Activation *activation;

    MatrixXf *weights;
    VectorXf *biases;

    /* 計算結果を格納する配列・変数*/
    MatrixXf *u;
    MatrixXf *z;
    MatrixXf *delta;

    /* 暗黙的なオブジェクトのコピーを回避 */
    Layer() = delete;
    Layer(Layer& other) = delete;
    Layer& operator=(const Layer& other) = delete;

    Layer(unsigned int n_in, unsigned int n_out, unsigned int n_data,
          Activation *activation, float learning_rate);

    void update();

public:
    virtual ~Layer();

    static Layer *newLayer(unsigned int n_in, unsigned int n_out,
                           unsigned int n_data, Activation *activation,
                           float learning_rate) {
        return new Layer(n_in, n_out, n_data, activation, learning_rate);
    }

    int getNIn() { return n_in; }

    int getNOut() { return n_out; }

    MatrixXf *getWeights() { return weights; }

    VectorXf *getBiases() { return biases; }

    MatrixXf *getU() { return u; }

    MatrixXf *getZ() { return z; }

    MatrixXf *getDelta() { return delta; }

    virtual MatrixXf *forward(MatrixXf *x);

    void backward(MatrixXf *nextDelta, MatrixXf *nextWeight);// 逆伝播関数

    virtual string toString();
};


#endif
