//
// Created by 金井廉 on 2016/06/21.
//

#ifndef CONVNETCPP_ACTIVATION_H
#define CONVNETCPP_ACTIVATION_H

#include <cmath>

float sigmoid(float x) {
    return 1. / (1. + exp(-x));
}

float g_sigmoid(float x) {
    float t = sigmoid(x);
    return t * (1. - t);
}

float relu(float x) {
    return (x > 0) ? x : 0;
}

float g_relu(float x) {
    return (x > 0) ? 1 : 0;
}

float iden(float x) {
    return x;
}

float g_iden(float x) {
    return 1;
}

#endif //CONVNETCPP_ACTIVATION_H
