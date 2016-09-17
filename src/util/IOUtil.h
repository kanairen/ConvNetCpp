//
// Created by Ren Kanai on 2016/09/15.
//

#ifndef CONVNETCPP_IOUTIL_H
#define CONVNETCPP_IOUTIL_H

#include <iostream>
#include <fstream>
#include <vector>

template<class T>
void print(const std::vector<T> &v) {
    for (T t: v) {
        std::cout << t << ", ";
    }
    std::cout << "\n";
}

template<class T>
void print_col(const std::vector<std::vector<T>> &v, int column_index) {
    for (int i = 0; i < v.size(); ++i) {
        std::cout << v[i][column_index] << ", ";
    }
    std::cout << "\n";
}

template<class T>
void save_as_csv(const std::string path, const std::vector<T> &v) {
    std::ofstream ofs(path);
    for (T t: v) {
        ofs << t << ",";
    }
    ofs << "\n";
    ofs.close();
}

#endif //CONVNETCPP_IOUTIL_H
