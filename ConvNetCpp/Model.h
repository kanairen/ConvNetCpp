#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include <algorithm>
#include <float.h>
#include "Layer.h"

using namespace std;

class Model{
private:
    // レイヤ配列
    vector<Layer*> *layers;
    
    // 順伝播後の予測格納コンテナ
    vector<int> *preds;
    // 誤差一時格納コンテナ
    vector<float> *delta;

    // コピーコンストラクタ及び代入演算子によるコピーの防止
    Model(const Model& model);
    Model& operator=(const Model& model);
    
    // 逆伝播
    void backward(vector<float>* delta);
public:
    Model();
    virtual ~Model();

    // レイヤの追加
    void addLayer(int n_in, int n_out, float learning_rate=0.001);
    // 順伝播
    vector<int>* forward(vector<vector<float>*>* inputs);
    vector<int>* forwardWithBackward(vector<vector<float>*> *inputs,vector<int> *answers);
    // 誤差
    static float error(vector<int>* predicts, vector<int>* answers);
    // 最大値インデックスのベクトル
    static int argmax(vector<float>* output);
};

#endif
