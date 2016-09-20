//
// Created by Ren Kanai on 2016/09/16.
//

#ifndef CONVNETCPP_OSUTIL_H
#define CONVNETCPP_OSUTIL_H

#include <stdio.h>
#include <string>
#include <vector>

using std::vector;
using std::string;

void list_dirs(string path, vector<string> &dst,
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
        std::cerr << "list_dirs() : failed to execute commands." <<
        std::endl;
        exit(1);
    }

    // 一時ファイル展開
    std::ifstream ifs(tmpLsPath);
    if (ifs.fail()) {
        std::cerr << "list_dirs() : failed to load labels." <<
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

#endif //CONVNETCPP_OSUTIL_H
