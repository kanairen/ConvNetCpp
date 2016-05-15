#include"Layer.h"

Layer::Layer(int n_in, int n_out, float learning_rate){
    this->prev = NULL;
    this->next = NULL;
    this->init(n_in, n_out,learning_rate);
}

Layer::Layer(Layer* prev, int n_out, float learning_rate){
    this->prev = prev;
    this->next = NULL;
    this->init(prev->n_out, n_out,learning_rate);
}

// メンバ初期化
void Layer::init(int n_in, int n_out,float learning_rate){
    
    // 入出力ユニット数初期化
    this->n_in = n_in;
    this->n_out = n_out;
    
    // 学習率
    this->learning_rate = learning_rate;
    
    // 乱数生成器
    float low = -sqrt(6./(n_in+n_out));
    float high = sqrt(6./(n_in+n_out));
    uniform_real_distribution<> real(low, high);
    mt19937 mt((random_device())());
    
    // パラメタ変数の初期化
    this->weights = new vector<vector<float>>(n_out);
    this->biases = new vector<float>(n_out);
    for(int i = 0; i < n_out; i++){
        (*this->biases)[i] = 0.0;
        (*this->weights)[i] = vector<float>(n_in);
        for(int j = 0; j < n_in; j++){
            (*this->weights)[i][j] = real(mt);
        }
    }
    
    this->u = new vector<float>(n_out);
    this->z = new vector<float>(n_out);
    this->delta = new vector<float>(n_out);
    this->b_delta = 0.0;
}

Layer::~Layer(){
    if (this->prev != NULL) {
        delete this->prev;
    }
    if (this->next != NULL) {
        delete this->next;
    }
    delete this->weights;
    delete this->biases;
    delete this->u;
    delete this->z;
    delete this->delta;
}

// 順伝播関数
// 伝播により、逆伝播に使う入力重み付き和uが求まる
vector<float>* Layer::forward(vector<float> *x){
    float u;
    for(int i = 0; i < this->n_out; i++){
        u = 0.0;
        for(int j = 0; j < this->n_in; j++){
            u += (*this->weights)[i][j] * (*x)[j];
        }
        u += (*this->biases)[i];
        (*this->u)[i] = u;
        (*this->z)[i] = Layer::activation(u);
    }
    if (this->next == NULL) {
        return this->z;
    }else{
        return this->next->forward(this->z);
    }
}

// 外部呼び出し用逆伝播関数
void Layer::backward(vector<float>* delta){
    if (this->next == NULL) {
        for (int i = 0; i < this->delta->size(); i++) {
            (*this->delta)[i] = (*delta)[i];
        }
        // パラメタ更新
        this->update();
        if (this->prev != NULL) {
            this->prev->backward();
        }
    }else{
        throw exception();
    }
}

// 逆伝播関数
// 出力層のデルタは、交差エントロピー誤差の場合、
// 推測結果と正解データの誤差になる
void Layer::backward(){
    if (this->next == NULL) {
        throw exception();
    }
    vector<float> *next_delta = this->next->getDelta();
    vector<vector<float>> *next_weight = this->next->getWeights();
    // 重みパラメタの逆伝播（次層のバイアスパラメタは現在層の重みパラメタから影響を受けないので、）
    for(int j  = 0; j < this->next->n_in; j++){
        (*this->delta)[j] = 0.0;
        for(int k = 0; k < this->next->n_out; k++){
            (*this->delta)[j] += (*next_delta)[k] * (*next_weight)[k][j];
        }
        (*this->delta)[j] *= Layer::gradActivation((*this->u)[j]);
    }
    // バイアスパラメタへの逆伝播
    this->b_delta = 0.0;
    for (int k = 0; k < this->next->n_out; k++) {
        this->b_delta += (*next_delta)[k] * 1.0;
    }
    
    // パラメタ更新
    this->update(); 
    if (this->prev != NULL) {
        this->prev->backward();
    }
}

// パラメタ更新関数
void Layer::update(){
    for(int i = 0; i < this->n_out; i++){
        for(int j = 0; j < this->n_in; j++){
            (*this->weights)[i][j] -= (*this->delta)[i] * this->learning_rate;
        }
        (*this->biases)[i] -= this->b_delta * this->learning_rate;
    }
}

// 活性化関数
float Layer::activation(float x){
    return x;
}

// 活性化関数微分形
float Layer::gradActivation(float x){
    return 1.0;
}

string Layer::toString(){
    stringstream ss;
    ss << "n_in : " << this->n_in << '\n';
    ss << "n_out : " << this->n_out << '\n';
    return ss.str();
}
