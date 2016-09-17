//
// Created by kanairen on 2016/06/15.
//

#include "MNIST.h"


int MNIST::toInteger(int i) {

    /*
     * 4バイト列を32bit整数に変換
     */

    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

void MNIST::loadData(std::string f_name, vector<vector<float>> &dst,
                     int &dst_n_row, int &dst_n_col) {

    /*
     * MNIST文字データをファイルから読み込み
     */

    cout << "load data : " << f_name << endl;

    ifstream ifs(f_name, std::ios::in | std::ios::binary);
    if (ifs.fail()) {
        cerr << "MNIST:ファイル読み込みエラー" << endl;
        exit(1);
    }

    int magic_number;
    int n_imgs;
    int n_row, n_col;

    ifs.read((char *) &magic_number, sizeof(magic_number));
    magic_number = MNIST::toInteger(magic_number);
    ifs.read((char *) &n_imgs, sizeof(n_imgs));
    n_imgs = MNIST::toInteger(n_imgs);
    ifs.read((char *) &n_row, sizeof(n_row));
    n_row = MNIST::toInteger(n_row);
    ifs.read((char *) &n_col, sizeof(n_col));
    n_col = MNIST::toInteger(n_col);

    dst_n_row = (unsigned int) n_row;
    dst_n_col = (unsigned int) n_col;

    std::cout << "magic number : " << magic_number << std::endl;
    std::cout << "number of images : " << n_imgs << std::endl;
    std::cout << "number of rows : " << n_row << std::endl;
    std::cout << "number of cols : " << n_col << std::endl;

    dst.resize(n_row * n_col);
    unsigned char p;
    for (int j = 0; j < n_imgs; ++j) {
        for (int i = 0; i < n_row * n_col; ++i) {
            if (j == 0) {
                dst[i].resize(n_imgs);
            }
            ifs.read((char *) &p, sizeof(p));
            dst[i][j] = (float) p / 255.f;
        }
    }

}

void MNIST::loadLabels(string f_name, vector<int> &dst) {

    /*
     * MNIST正解データをファイルから読み込み
     */

    cout << "load labels : " << f_name << endl;

    ifstream ifs(f_name, std::ios::in | std::ios::binary);
    if (ifs.fail()) {
        cerr << "MNIST:ファイル読み込みエラー" << endl;
        exit(1);
    }

    int magic_number;
    int n_labels;

    ifs.read((char *) &magic_number, sizeof(magic_number));
    magic_number = MNIST::toInteger(magic_number);
    ifs.read((char *) &n_labels, sizeof(n_labels));
    n_labels = MNIST::toInteger(n_labels);

    cout << "magic number : " << magic_number << endl;
    cout << "number of labels : " << n_labels << endl;

    dst.resize(n_labels);

    unsigned char p;
    for (int i = 0; i < n_labels; i++) {
        ifs.read((char *) &p, sizeof(p));
        dst[i] = (int) p;
    }
}
