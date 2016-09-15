//
// Created by Ren Kanai on 2016/09/15.
//

#ifndef CONVNETCPP_IOUTIL_H
#define CONVNETCPP_IOUTIL_H

#include <iostream>
#include <fstream>
#include <vector>

template<class T>
void output_stream(const std::ostream ost, const vector <T> &v,
                   const char delim) {
    for (T t: v) {
        ost << t << delim;
    }
    ost << "\n";
}

template<class T>
void print(const vector <T> &v) {
    output_stream(std::cout, v, ' ');
}

template<class T>
void save_as_csv(const string path, const vector <T> &v) {
    std::ofstream ofs(path);
    output_stream(ofs, v, ',');
    ofs.close();
}

#endif //CONVNETCPP_IOUTIL_H
