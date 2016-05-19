//
//  main.cpp
//  ConvNetCpp
//
//  Created by 金井廉 on 2016/05/07.
//  Copyright © 2016年 金井廉. All rights reserved.
//

#include "test.h"


int main(int argc, const char * argv[]) {
    if(argc == 0){
        cerr << "error : コマンドライン引数にパスを設定してください。" << endl;
        exit(1);
    }
    string filedir = argv[1];
    mnist(filedir);
    return 0;
}

