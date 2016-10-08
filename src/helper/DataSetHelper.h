//
// Created by Ren Kanai on 2016/10/08.
//

#ifndef CONVNETCPP_DATASETHELPER_H
#define CONVNETCPP_DATASETHELPER_H

#include "../Data.h"

class DataSetHelper {
private:

    DataSetHelper() = delete;

    DataSetHelper(const DataSetHelper &dataset_helper) = delete;

    virtual ~DataSetHelper() = default;

public:

    enum Type : long {
        MNIST_ = 0x00,
        BAND_SHAPE_MAP = 0x01,
    };

    static DataSet<float, int> *get_dataset(int id, char *argv[]) {
        switch (id) {
            case MNIST_:
                return new MNIST(argv[2], argv[3], argv[4], argv[5]);
            case BAND_SHAPE_MAP:
                return new ShapeMapSet(argv[2], argv[3]);
            default:
                error_and_exit(
                        "DataSetHelper::get_dataset : failed to get dataset.");
                return nullptr;
        }
    }

};

#endif //CONVNETCPP_DATASETHELPER_H
