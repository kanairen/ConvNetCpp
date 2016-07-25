//
// Created by kanairen on 2016/06/14.
//

#include "TestLayer.h"
#include "TestSoftMaxLayer.h"
#include "TestGridLayer.h"
#include "TestConvLayer.h"
#include "TestMaxPoolLayer.h"

#include "TestModel.h"

int main() {

    // test Layer
    layer::test_init();
    layer::test_forward();
    layer::test_backward();

    // test Softmax Layer
    sm_layer::test_init();
    sm_layer::test_forward();
    sm_layer::test_backward();

    // test Grid Layer
    grid_layer::test_init();
    grid_layer::test_filter_outsize();

    // test Convolution Layer
    conv_layer::test_init();
    conv_layer::test_backward();

    // test MaxPooling Layer
    max_pool_layer::test_init();
    max_pool_layer::test_forward();
    max_pool_layer::test_backward();

    // test Model
    model::test_init();
    model::test_forward();
    model::test_backward();


}
