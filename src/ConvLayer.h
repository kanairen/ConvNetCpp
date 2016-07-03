//
// Created by kanairen on 2016/06/29.
//

#ifndef CONVNETCPP_CONVLAYER_H
#define CONVNETCPP_CONVLAYER_H

#include <random>
#import "AbstractLayer.h"

class ConvLayer2d : public AbstractLayer {
private:

    static const int T_WEIGHT_DISABLED = -1;

    // 成分数 (フィルタ幅)^2 × (入力チャネル) × （出力チャネル） のフィルタベクトル
    vector<float> h;
    // 入力画素(i, j)が時刻rにおいて、フィルタと重なるかどうかを保持するベクトル
    // 軸は(r, i, j)とする
    vector<vector<int>> t;

    unsigned int input_width;
    unsigned int input_height;

    unsigned int c_in;
    unsigned int c_out;

    unsigned int kw;
    unsigned int kh;

    unsigned int stride;

    void update(const vector<vector<float>> &prev_output,
                const float learning_rate);

public:

    ConvLayer2d(unsigned int n_data,
                unsigned int input_width, unsigned int input_height,
                unsigned int c_in, unsigned int c_out,
                unsigned int kw, unsigned int kh, unsigned int stride,
                float (*activation)(float), float (*grad_activation)(float));

    const vector<vector<float>> &get_weights();


};

#endif //CONVNETCPP_CONVLAYER_H
