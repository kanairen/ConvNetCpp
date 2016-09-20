//
// Created by Ren Kanai on 2016/09/16.
//

#ifndef CONVNETCPP_SHAPEMAPDATASET_H
#define CONVNETCPP_SHAPEMAPDATASET_H

#include "BaseDataSet.h"

class ShapeMapDataSet : public BaseDataSet {
public:
    static const unique_ptr<ShapeMapDataSet> load(std::string path) {

    }

};

template<class T>
class BaseShapeMap {

    /*
     * 基底形状マップクラス
     */

protected:
    // 元々の3Dモデルを識別するためのID
    unsigned int model_id;
    // クラスラベル
    unsigned int cls;
    // 辺の分割数
    unsigned int n_div;
    // 走査方向
    std::string direction;
    // データ型
    std::string data_type;
    // 3Dモデルの重心から表面までの距離
    unique_ptr<std::vector<T>> distances;
    // 各行のカラム数
    unique_ptr<std::vector<unsigned int>> n_columns;

    BaseShapeMap(unsigned int model_id, unsigned int cls, unsigned int n_div,
                 std::string direction, std::string data_type,
                 unique_ptr<std::vector<T>> distances,
                 unique_ptr<std::vector<unsigned int>> n_columns)
            : model_id(model_id), cls(cls), n_div(n_div),
              direction(direction), data_type(data_type),
              distances(distances), n_columns(n_columns) { }

    virtual ~BaseShapeMap() = default;

};

template<class T>
class BandShapeMap : public BaseShapeMap<T> {

    /*
     * 帯型形状マップクラス
     */

public:

    static unique_ptr<BaseShapeMap> load(std::string file_path) {

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

        unsigned int model_id;
        unsigned int cls;
        unsigned int n_div;
        std::string direction;
        std::string data_type;
        unique_ptr<std::vector<T>> distances;
        unique_ptr<std::vector<unsigned int>> n_columns;

        // # 各マップのカラム数を読み込み
        string line;
        while (getline(ifs, line) and line.find("#DATA")) {
            if (line == "#ID") {
                getline(ifs, line);
                model_id = atoi(line.c_str());
            } else if (line == "#CLASS") {
                getline(ifs, line);
                cls = atoi(line.c_str());
            } else if (line == "#N_DIV") {
                getline(ifs, line);
                n_div = atoi(line.c_str());
            } else if (line == "#DIRECTION") {
                getline(ifs, line);
                direction = line;
            } else if (line == "#DATA_TYPE") {
                getline(ifs, line);
                type = line;
            } else {
                continue;
            }
        }

        // 各マップのデータ部を読み込み
        distances = new std::vector<float>();
        float f;
        while (ifs.read((char *) &f, sizeof(float))) {
            distances.push_back(f);
        }

        // 各行のカラム数
        n_columns = new std::vector<int>();
        for (int row = 0; row <= n_div; ++row) {
            row_size.push_back(distances.size() / n_div);
        }

        return unique_ptr<BaseShapeMap>(
                new BaseShapeMap(model_id, cls, n_div, direction, data_type,
                                 distances, n_columns));

    }
};

template<class T>
class UniShapeMap : public BaseShapeMap<T> {
private:

    static unique_ptr<vector<T>> read_distances(const std::ifstream &ifs,
                                                vector<unsigned int> &n_columns) {

        unique_ptr<vector<T>> distances = new vector<T>();

        // 各マップのデータ部を読み込み
        T f;
        for (int row = 0; row <= n_div; ++row) {
            n_columns.push_back(row + 1);
            for (int i = 0; i <= row; ++i) {
                ifs.read((char *) &f, sizeof(T));
                distances.push_back(f);
            }
        }

        return distances;
    }

protected:
    unsigned int face_id;

};

#endif //CONVNETCPP_SHAPEMAPDATASET_H
