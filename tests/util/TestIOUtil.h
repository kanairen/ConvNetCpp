//
// Created by Ren Kanai on 2016/09/15.
//

#ifndef CONVNETCPP_TESTIOUTIL_H
#define CONVNETCPP_TESTIOUTIL_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <gtest/gtest.h>
#include "../ShareArgs.h"
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

    std::stringbuf s_buf;
    std::streambuf *prev_buf = std::cout.rdbuf(&s_buf);

    print(v);

    std::cout.rdbuf(prev_buf);

    std::string result = "";
    for (int elem : v) {
        result += std::to_string(elem) + ", ";
    }
    result += "\n";

    ASSERT_EQ(s_buf.str(), result);

}

TEST_F(IOUtilTest, test_print_col) {
    std::cout << "IOUtilTest::test_print()" << std::endl;

    std::stringbuf s_buf;
    std::streambuf *prev_buf = std::cout.rdbuf(&s_buf);

    std::vector<std::vector<int>> v = {{1, 2, 3},
                                       {4, 5, 6}};

    int col = 0;

    print_col(v, col);

    std::cout.rdbuf(prev_buf);

    std::string result = "";
    for (const std::vector<int> &v_row : v) {
        for (int i = 0; i < v_row.size(); ++i) {
            if (i == col) {
                result += std::to_string(v_row[i]) + ", ";
            }
        }
    }
    result += "\n";

    ASSERT_EQ(s_buf.str(), result);

}

TEST_F(IOUtilTest, test__save_as_csv) {
    std::string path = std::string(ARGV[1]) + "/test_io_util.csv";
    std::cout << "IOUtilTest::test_save_as_csv()" << std::endl;
    save_as_csv(path, v);

    std::ifstream ifs(path);
    std::string line;
    while (getline(ifs, line)) {
        std::cout << line << std::endl;
    }

    remove(path.c_str());
}

#endif //CONVNETCPP_TESTIOUTIL_H
