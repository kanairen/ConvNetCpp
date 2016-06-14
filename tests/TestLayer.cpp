//
// Created by kanairen on 2016/06/14.
//

#include "TestLayer.h"

void test_layer() {

    // 関数
    float identity(float x) {
        return x;
    }

    // 入力データ
    vector<vector<float>> v(5, vector<float>(10, 1));

    // Layerオブジェクト
    Layer l(5, 10, 10, identity);

    // 順伝播
    const vector<vector<float>> &output = l.forward(v);


}