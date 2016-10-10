//
// Created by Ren Kanai on 2016/10/01.
//

#ifndef CONVNETCPP_STRINGUTIL_H
#define CONVNETCPP_STRINGUTIL_H

bool atob(const char c[]) {
    if (std::strncmp(c, "true", 4) == 0) {
        return true;
    } else if (std::strncmp(c, "false", 5) == 0) {
        return false;
    } else {
        error_and_exit("atob() : failed to convert an argument to boolean.");
    }
    return false;
}

bool is_equal(const char s1[],const char s2[]){
    return std::strcmp(s1, s2) == 0;
}

#endif //CONVNETCPP_STRINGUTIL_H
