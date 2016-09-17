//
// Created by Ren Kanai on 2016/09/16.
//

#ifndef CONVNETCPP_TESTOSUTIL_H
#define CONVNETCPP_TESTOSUTIL_H

#include <string>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../ShareArgs.h"
#include "../../src/util/OSUtil.h"

class OSUtilTest : public ::testing::Test {
protected:

    OSUtilTest() { }

    virtual ~OSUtilTest() { }

    virtual void SetUp() {
        std::cout << "OSUtilTest::SetUp()" << std::endl;
    }

    virtual void TearDown() {
        std::cout << "OSUtilTest::TearDown()" << std::endl;
    }

};

TEST_F(OSUtilTest, test_list_dirs) {
    std::cout << "OSUtilTest::listdir()" << std::endl;
    std::vector<std::string> v;
    list_dirs(PATH_ROOT, v);
    for (std::string s : v) {
        std::cout << s << std::endl;
    }
}

#endif //CONVNETCPP_TESTOSUTIL_H
