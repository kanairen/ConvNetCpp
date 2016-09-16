//
// Created by Ren Kanai on 2016/09/16.
//

#ifndef CONVNETCPP_IMAGEDATASET_H
#define CONVNETCPP_IMAGEDATASET_H

#include "BaseDataSet.h"

template<class X, class Y>
class ImageDataSet : public BaseDataSet<X, Y> {
public:
    unsigned int width;
    unsigned int height;
};


#endif //CONVNETCPP_IMAGEDATASET_H
