//
// Created by kanairen on 2016/06/14.
//

#include "../src/config.h"

#ifndef CONV_NET_CPP_DEBUG

#include "TestMNIST.h"
#include "TestLayer.h"

int main(){
    test_mnist();
    test_layer();
}

#endif
