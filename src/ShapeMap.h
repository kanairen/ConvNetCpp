//
// Created by kanairen on 2016/08/25.
//

#ifndef CONVNETCPP_SHAPEMAP_H
#define CONVNETCPP_SHAPEMAP_H

#include <stdio.h>
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
        while (getline(ifs, line) and line.find("#DATA")) {
            if (line == "#CLASS") {
                getline(ifs, line);
                cls = atoi(line.c_str());
            } else if (line == "#FACE_ID") {
                getline(ifs, line);
                face_id = atoi(line.c_str());
            } else if (line == "#DIRECTION") {
                getline(ifs, line);
                direction = line;
            } else if (line == "#N_DIV") {
                getline(ifs, line);
                n_div = atoi(line.c_str());
            } else if (line == "#TYPE") {
                getline(ifs, line);
                type = line;
            } else {
                continue;
            }
        }

        // 各マップのデータ部を読み込み
        float f;

        for (int row = 0; row <= n_div; ++row) {
            row_size.push_back(row + 1);
            for (int i = 0; i <= row; ++i) {
                ifs.read((char *) &f, sizeof(float));
                distances.push_back(f);
            }
        }

    }

public:

    unsigned int face_id;

    unsigned int cls;

    unsigned int n_div;

    string direction;

    string type;

    vector<float> distances;

    vector<unsigned int> row_size;

    ShapeMap(string file_path) {
        loadMap(file_path);
    }

    virtual ~ShapeMap() { }

};

// << operator
std::ostream &operator<<(std::ostream &os, const ShapeMap &map) {
    os << "class : " << map.cls << "  face ID : " << map.face_id <<
    "  direction : " << map.direction << "\n";
    int index = 0;
    for (unsigned int rs: map.row_size) {
        os << "[ ";
        for (int i = 0; i < rs; ++i) {
            os << map.distances[index++] << " ";
        }
        os << "]\n";
    }
    return os;
}

class ShapeMapSet {
private:

    void loadData(string root, vector<ShapeMap> &dst) {

        /*
         *
         * データファイルのルートディレクトリを指定し、データメンバを初期化する
         *
         * root : データファイルのルートディレクトリ
         * dst  : 出力コンテナ
         *
         */

        vector<string> label_names;
        vector<string> direction_names;
        vector<string> file_names;

        listDirs(root, label_names);
        for (string label_name : label_names) {
            string label_path = root + "/" + label_name;
            direction_names.clear();
            listDirs(label_path, direction_names);
            for (string direction_name : direction_names) {
                string direction_path = label_path + "/" + direction_name;
                file_names.clear();
                listDirs(direction_path, file_names);
                for (string file_name: file_names) {
                    string full_path = direction_path + "/" + file_name;

                    dst.push_back(ShapeMap(full_path));

                }
            }
        }


    }

    void listDirs(string path, vector<string> &dst,
                  string tmp_file_name = "tmp_ls_path.txt") {

        /**
         *
         *  引数にとったディレクトリ中のファイル/フォルダ名一覧を文字列として返す
         *
         *  path : 対象ディレクトリ
         *
         */

        // ファイル/フォルダ一覧を一時保存するためのファイル生成
        string tmpLsPath = path + "/" + tmp_file_name;
        string commands = "ls " + path + " >> " + tmpLsPath;
        if (std::system(commands.c_str())) {
            std::cerr << "ShapeMap::listDirs() : failed to execute commands." <<
            std::endl;
            exit(1);
        }

        // 一時ファイル展開
        std::ifstream ifs(tmpLsPath);
        if (ifs.fail()) {
            std::cerr << "ShapeMap::listDirs() : failed to load labels." <<
            std::endl;
            exit(1);
        }

        // 一時ファイル中のファイル/フォルダ名をdstに格納
        string line;
        while (getline(ifs, line)) {
            int isNotFound = line.find(tmp_file_name);
            if (!isNotFound) {
                continue;
            }
            dst.push_back(line);
        }

        // 一時ファイルの消去
        std::remove(tmpLsPath.c_str());
    }

public:

    vector<ShapeMap> train_maps;
    vector<ShapeMap> test_maps;

    ShapeMapSet(string root_train, string root_test) {
        loadData(root_train, train_maps);
        loadData(root_test, test_maps);
    };

    ~ShapeMapSet() { };

    unsigned int xv_size() {
        return 0;
    }

};


#endif //CONVNETCPP_SHAPEMAP_H
