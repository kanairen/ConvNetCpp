//
// Created by kanairen on 2016/06/14.
//

#include "TestLayer.h"

// 関数
float identity(float x) {
    return x;
}

float grad_identity(float x) {
    return 1;
}

void test_layer() {

    // 入力データ
    vector<vector<float>> v(5, vector<float>(10, 1));

    // Layerオブジェクト
    Layer l(10, 5, 10, identity, grad_identity);

    // 順伝播
    const vector<vector<float>> &output = l.forward(v);

}