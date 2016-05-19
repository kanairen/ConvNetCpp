//
//  MNIST.cpp
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/14.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#include "MNIST.h"

MNIST::MNIST(string fname_x_train,
             string fname_x_test,
             string fname_y_train,
             string fname_y_test,
             unsigned int n_batch){
    this->x_train = this->loadData(fname_x_train, n_batch);
    this->x_test = this->loadData(fname_x_test, n_batch);
    this->y_train = this->loadLabels(fname_y_train, n_batch);
    this->y_test = this->loadLabels(fname_y_test, n_batch);
}

MNIST::~MNIST(){
    // TODO more better expression
    for (int i = 0; i < this->x_train->size(); i++){
        for (int j = 0; j < (*this->x_train)[i]->size(); j++) {
            delete (*(*this->x_train)[i])[j];
        }
         delete (*this->x_train)[i];
    }
    delete this->x_train;
    
    for (int i = 0; i < this->x_test->size(); i++){
        for (int j = 0; j < (*this->x_test)[i]->size(); j++) {
            delete (*(*this->x_test)[i])[j];
        }
         delete (*this->x_test)[i];
    }
    delete this->x_test;
    
    for (int i = 0; i < this->y_train->size(); i++) {
        delete (*this->y_train)[i];
    }
    delete this->y_train;
    
    for (int i = 0; i < this->y_test->size(); i++) {
        delete (*this->y_test)[i];
    }
    delete this->y_test;
}

// 4バイト列を32bit整数に変換
int MNIST::toInteger(int i){
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// ヒープ上vectorオブジェクトへの参照は内部で生成する
vector<vector<vector<float>*>*>* MNIST::loadData(string filename, unsigned int n_batch){
    
    cout << "load data : " << filename << endl;
    
    ifstream ifs(filename,ios::in|ios::binary);
    if(ifs.fail()){
        cerr << "MNIST:ファイル読み込みエラー" << endl;
        exit(1);
    }
    
    int magic_number;
    int n_imgs;
    int n_row, n_col;
    
    ifs.read((char*)&magic_number, sizeof(magic_number));
    magic_number = MNIST::toInteger(magic_number);
    ifs.read((char*)&n_imgs, sizeof(n_imgs));
    n_imgs = MNIST::toInteger(n_imgs);
    ifs.read((char*)&n_row, sizeof(n_row));
    n_row = MNIST::toInteger(n_row);
    ifs.read((char*)&n_col, sizeof(n_col));
    n_col = MNIST::toInteger(n_col);
    
    cout << "magic number : " << magic_number << endl;
    cout << "number of images : " << n_imgs << endl;
    cout << "number of rows : " << n_row << endl;
    cout << "number of cols : " << n_col << endl;
    
    if(n_imgs % n_batch != 0){
        cerr << "error : 入力データがバッチ数で割りきれません" << endl;
        exit(1);
    }
    
    vector<vector<vector<float>*>*>* v = new vector<vector<vector<float>*>*>(n_batch);
    
    unsigned char p;
    for (int i = 0; i < n_batch; i++){
        (*v)[i] = new vector<vector<float>*>(n_imgs / n_batch);
        for (int j = 0; j < n_imgs / n_batch; j++) {
            (*(*v)[i])[j] = new vector<float>(n_row * n_col);
            
            for (int row = 0; row < n_row; row++) {
                for (int col = 0; col < n_col; col++) {
                    ifs.read((char*)&p, sizeof(p));
                    (*(*(*v)[i])[j])[row * n_col + col] = (float)p;
                }
            }
            
        }
    }
    return v;
}

// ヒープ上vectorオブジェクトへの参照は内部で生成する
vector<vector<int>*>* MNIST::loadLabels(string filename, unsigned int n_batch){
    
    cout << "load labels : " << filename << endl;
    
    ifstream ifs(filename,ios::in|ios::binary);
    if(ifs.fail()){
        cerr << "MNIST:ファイル読み込みエラー" << endl;
        exit(1);
    }
    
    int magic_number;
    int n_labels;
    
    ifs.read((char*)&magic_number, sizeof(magic_number));
    magic_number = MNIST::toInteger(magic_number);
    ifs.read((char*)&n_labels, sizeof(n_labels));
    n_labels = MNIST::toInteger(n_labels);
    
    cout << "magic number : " << magic_number << endl;
    cout << "number of labels : " << n_labels << endl;
    
    if(n_labels % n_batch != 0){
        cerr << "error : 入力データがバッチ数で割りきれません" << endl;
        exit(1);
    }
    
    vector<vector<int>*> *v = new vector<vector<int>*>(n_labels);
    
    unsigned char p;
    for (int i = 0; i < n_batch; i++) {
        (*v)[i] = new vector<int>(n_labels / n_batch);
        for (int j = 0; j < n_labels / n_batch; j++) {
            ifs.read((char*)&p, sizeof(p));
            (*(*v)[i])[j] = (int)p;
        }
    }
    return v;
}
