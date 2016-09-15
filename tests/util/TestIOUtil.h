//
// Created by Ren Kanai on 2016/09/15.
//

#ifndef CONVNETCPP_TESTIOUTIL_H
#define CONVNETCPP_TESTIOUTIL_H

#include "stdio.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "../../src/util/IOUtil.h"

class IOUtilTest : public ::testing::Test {
protected:

    std::vector<int> v;

    IOUtilTest() { }

    virtual ~IOUtilTest() { }

    virtual void SetUp() {
        std::cout << "IOUtilTest::SetUp()" << std::endl;
        v = {1, 2, 3};
    }

    virtual void TearDown() {
        std::cout << "IOUtilTest::TearDown()" << std::endl;
    }

};

TEST_F(IOUtilTest, test_print) {
    std::cout << "IOUtilTest::test_print()" << std::endl;
    print(v);
}

TEST_F(IOUtilTest, test__save_as_csv) {
    std::string path = "test_io_util.csv";
    std::cout << "IOUtilTest::test_save_as_csv()" << std::endl;
    save_as_csv(path, v);

    std::ifstream ifs(path);
    std::string line;
    while(getline(ifs, line)){
        std::cout << line << std::endl;
    }

    remove(path.c_str());
}

#endif //CONVNETCPP_TESTIOUTIL_H
