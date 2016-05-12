#ifndef LAYER_H_
#define LAYER_H_

#include <stdio.h>
#include <sstream>
#include <random>

using namespace std;

class Layer{
private:
    // 隣接レイヤへのポインタ
    Layer* prev;
    Layer* next;
    
    // 入出力ユニット数
    float n_in, n_out;
    // 重み変数 行数はn_out, 列数はn_in
    vector<vector<float>> *weights;
    // バイアス変数
    vector<float> *biases;
    
    /* 結果を格納するコンテナ*/
    // 入力の重み付き和
    vector<float> *u;
    // 順伝播出力
    vector<float> *z;
    // 逆伝播デルタ
    vector<float> *delta;
    // バイアス対応デルタ
    float b_delta;
    
    float learning_rate;
    
    // デフォルトコンストラクタを明示的に利用不可に
    Layer();
    // コピーコンストラクタの外部からの呼び出しを不可に
    Layer(const Layer& otherLayer);
    // 代入演算子を利用不可に
    Layer& operator=(const Layer& otherLayer);
    
    // メンバ初期化関数
    void init(int n_in, int n_out, float learning_rate);
    // 逆伝播関数
    void backward();
    // パラメタ更新関数
    void update();
    
public:
    Layer(int n_in, int n_out, float learning_rate = 0.001);
    Layer(Layer* prev, int n_out, float learning_rate = 0.001);
    virtual ~Layer();
    
    vector<vector<float>>* getWeights();
    vector<float>* getBiases();
    vector<float>* getDelta();
    
    // 順伝播関数
    vector<float>* forward(vector<float> *x);
    // 外部呼び出し用逆伝播関数
    void backward(vector<float>* delta);
    // パラメタ更新
    void update(vector<float> *delta);
    // 活性化関数
    virtual float activation(float x);
    // 活性化関数微分形
    virtual float gradActivation(float x);
    
    virtual string toString();
};

#endif
