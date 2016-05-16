//
//  Activation.h
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/15.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

class Activation{
public:
    virtual float f(float x) = 0;
    virtual float gf(float x) = 0;
};

class Sigmoid: public Activation{
public:
    virtual float f(float x){return 1. / (1. + exp(-x));}
    virtual float gf(float x){return (1. - this->f(x)) * this->f(x);}
};

class ReLU: public Activation{
public:
    virtual float f(float x){return x * (x > 0);}
    virtual float gf(float x){return (float)(x > 0);}
};

#endif /* ACTIVATION_H */
