//
// Created by kanairen on 2016/08/25.
//

#ifndef CONVNETCPP_SHAPEMAP_H
#define CONVNETCPP_SHAPEMAP_H

#include <iostream>
#include <fstream>

using std::vector;
using std::string;

class ShapeMap {
private:

    void loadMap(string file_path) {

        /*
         * 一つの距離マップを読み込み、ShapeMapメンバを更新する
         *
         * file_path : 距離マップファイルパス
         */

        std::ifstream ifs(file_path);
        if (ifs.fail()) {
            std::cerr <<
            "ShapeMap::loadData() : failed to load shape map file." <<
            std::endl;
            exit(1);
        }

        // # 各マップのカラム数を読み込み
        string line;
        vector<int> *n_column;
        while (getline(ifs, line) and line.find("#DATA")) {
            if (line == "#N_COLUMN HORIZON") {
                n_column = &n_column_horizon;
            } else if (line == "#N_COLUMN LOWER") {
                n_column = &n_column_lower;
            } else if (line == "#N_COLUMN UPPER") {
                n_column = &n_column_upper;
            } else if (line[0] == '#') {
                continue;
            }
            n_column->push_back(atoi(line.c_str()));
        }

        // 各マップのデータ部を読み込み
        float f;

        // horizontal map
        for (int nc : n_column_horizon) {
            for (int i = 0; i < nc; ++i) {
                ifs.read((char *) &f, sizeof(float));
                horizontal_map.push_back(f);
            }
        }

        // lower map
        for (int nc : n_column_lower) {
            for (int i = 0; i < nc; ++i) {
                ifs.read((char *) &f, sizeof(float));
                lower_map.push_back(f);
            }
        }

        // upper map
        for (int nc : n_column_upper) {
            for (int i = 0; i < nc; ++i) {
                ifs.read((char *) &f, sizeof(float));
                upper_map.push_back(f);
            }
        }
    }

public:

    vector<float> horizontal_map;
    vector<float> lower_map;
    vector<float> upper_map;

    vector<int> n_column_horizon;
    vector<int> n_column_lower;
    vector<int> n_column_upper;

    ShapeMap(string file_path) {
        loadMap(file_path);
    }

    virtual ~ShapeMap() { }

};


class ShapeMapSet {
private:

    void loadData(string name_list_path, string x_data_root_path,
                  vector<ShapeMap> &dst) {
        /*
         * name_list_path   : 各データファイル名をまとめたファイルへのパス
         * x_data_root_path : 実際のデータファイルのルートパス
         * dst              : 出力コンテナ
         */

        std::ifstream ifs(name_list_path);
        if (ifs.fail()) {
            std::cerr <<
            "ShapeMap::loadData() : failed to load name list file." <<
            std::endl;
            exit(1);
        }

        // リストからパスを読み込み、配列に格納していく
        string line;
        while (getline(ifs, line)) {
            dst.push_back(ShapeMap(x_data_root_path + "/" + line));
        }

    }

    void loadLabels(string label_path, vector<int> &dst) {

        /*
         * label_path : 正解データラベル一覧ファイルへのパス
         * dst        : 出力コンテナ
         */

        std::ifstream ifs(label_path);
        if (ifs.fail()) {
            std::cerr << "ShapeMap::loadLabels() : failed to load labels." <<
            std::endl;
            exit(1);
        }

        string line;
        while (getline(ifs, line)) {
            dst.push_back(atoi(line.c_str()));
        }

    }


public:

    vector<ShapeMap> x_train;
    vector<ShapeMap> x_test;
    vector<int> y_train;
    vector<int> y_test;

    ShapeMapSet(string x_train_name_list_path, string x_test_name_list_path,
                string y_train_path, string y_test_path,
                string x_data_root_path) {
        loadData(x_train_name_list_path, x_data_root_path, x_train);
        loadData(x_test_name_list_path, x_data_root_path, x_test);
        loadLabels(y_train_path, y_train);
        loadLabels(y_test_path, y_test);
    };

    ~ShapeMapSet() { };

    unsigned int xv_size() {
        return 0;
    }

};


#endif //CONVNETCPP_SHAPEMAP_H
