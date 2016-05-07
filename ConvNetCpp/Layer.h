#ifndef LAYER_H_
#define LAYER_H_

#include <stdio.h>
#include <sstream>
#include <random>

using namespace std;

class Layer{
private:
    // 入出力ユニット数
    float n_in, n_out;
    // 重み変数 行数はn_out, 列数はn_in
    vector<vector<float>> *weights;
    // バイアス変数
    vector<float> *biases;

    // 入力の重み付き和
    vector<float> *u;
    // 順伝播出力
    vector<float> *z;
    // 逆伝播デルタ
    vector<float> *delta;

    // デフォルトコンストラクタを明示的に無効化
    Layer();
    // コピーコンストラクタを無効化
    Layer(const Layer& otherLayer);
    // 代入演算子の無効化
    Layer& operator=(const Layer& otherLayer);

public:
    // 引数付きコンストラクタ
    Layer(int n_in, int n_out);
    // 具象クラスのデストラクタが必ず呼ばれるようにvirtual化
    virtual ~Layer();
    
    // アクセサ
    vector<vector<float>>* getWeights();
    vector<float>* getBiases();
    vector<float>* getDelta();

    // 順伝播関数
    vector<float>* forward(vector<float> *x);
    // 逆伝播関数
    vector<float>* backward(vector<float> *nextDelta, vector<vector<float>> *nextWeight);
    // パラメタ更新
    void update(vector<float> *delta);
    // 活性化関数
    virtual float activation(float x);
    // 活性化関数微分形
    virtual float gradActivation(float x);

    virtual string toString();
};

#endif
