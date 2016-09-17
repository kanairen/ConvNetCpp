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

    void load(string file_path) {

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
            if (line == "#ID"){
                getline(ifs, line);
                id = atoi(line.c_str());
            } else if (line == "#CLASS") {
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
            } else if (line == "#DATA_TYPE") {
                getline(ifs, line);
                type = line;
            } else {
                continue;
            }
        }

        // 余分なバイナリを空読み
        getline(ifs,line);
        getline(ifs,line);

        // 各マップのデータ部を読み込み
        float f;
        while (ifs.read((char *) &f, sizeof(float))) {
            distances.push_back(f);
        }

        if (distances.size() != (5 * n_div) * (n_div + 1)) {
            std::cerr << "The number of distances are not correct." <<
            std::endl;
            exit(1);
        }

        // 各行のカラム数
        for (int row = 0; row <= n_div; ++row) {
            row_size.push_back(distances.size() / n_div);
        }

    }

public:

    int id;

    int face_id;

    int cls;

    int n_div;

    string direction;

    string type;

    vector<float> distances;

    vector<int> row_size;

    ShapeMap(string file_path) {
        load(file_path);
    }

    virtual ~ShapeMap() { }

    int data_size() {
        return distances.size();
    }
};

// << operator
std::ostream &operator<<(std::ostream &os, const ShapeMap &map) {
    os << "class : " << map.cls << "  face ID : " << map.face_id <<
    "  direction : " << map.direction << "\n";
    int index = 0;
    for (int rs: map.row_size) {
        os << "[ ";
        for (int i = 0; i < rs; ++i) {
            os << map.distances[index++] << " ";
        }
        os << "]\n";
    }
    return os;
}

// Eigenがunsigned int コンテナを持たない（？）ため、int
class ShapeMapSet : public DataSet<float, int> {
private:

    void load(string root, vector<ShapeMap> &dst_maps,
              vector<vector<float>> &dst_x, vector<int> &dst_y) {

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
            std::cout << label_name << std::endl;
            string label_path = root + "/" + label_name;
            direction_names.clear();
            listDirs(label_path, direction_names);
            for (string direction_name : direction_names) {
                string direction_path = label_path + "/" + direction_name;
                file_names.clear();
                listDirs(direction_path, file_names);
                for (string file_name: file_names) {
                    string full_path = direction_path + "/" + file_name;

                    dst_maps.push_back(ShapeMap(full_path));

                }
            }
        }

        if (dst_maps.size() == 0) {
            return;
        }

        dst_x.resize(dst_maps[0].data_size());
        dst_y.resize(dst_maps.size());

        // データ部への距離のコピー
        for (int dx_col = 0; dx_col < dst_maps.size(); ++dx_col) {
            const vector<float> &dists = dst_maps[dx_col].distances;
            for (int dx_row = 0; dx_row < dists.size(); ++dx_row) {
                dst_x[dx_row].push_back(dists[dx_row]);
            }
        }

        // クラスラベル集合
        std::set<int> set_y;
        for (int i = 0; i < dst_maps.size(); ++i) {
            set_y.insert(dst_maps[i].cls);
        }
        // クラスラベル空間
        for (int i = 0; i < dst_maps.size(); ++i) {
            auto iter = set_y.find(dst_maps[i].cls);
            if (iter == set_y.end()) {
                cerr << "ShapeMapSet::load() : failed to find label." << endl;
                exit(1);
            }
            dst_y[i] = std::distance(set_y.begin(), iter);
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
        load(root_train, train_maps, x_train, y_train);
        load(root_test, test_maps, x_test, y_test);
    };

    ~ShapeMapSet() { };

    int data_size() {
        if (train_maps.size() > 0) {
            return train_maps[0].data_size();
        } else if (test_maps.size() > 0) {
            return test_maps[0].data_size();
        } else {
            std::cerr << "data is not exists." << std::endl;
            return -1;
        }
    }

};


#endif //CONVNETCPP_SHAPEMAP_H
