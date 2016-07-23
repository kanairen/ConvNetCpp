//
// Created by kanairen on 2016/06/14.
//

#include "TestLayer.h"
#include "TestSoftMaxLayer.h"

int main() {

    // test Layer
    layer::test_init();
    layer::test_forward();
    layer::test_backward();

    // test softmax Layer
    sm_layer::test_init();
}
