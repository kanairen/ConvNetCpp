//
// Created by Ren Kanai on 2016/10/08.
//

#ifndef CONVNETCPP_ACTIVATIONHELPER_H
#define CONVNETCPP_ACTIVATIONHELPER_H

#include "../activation.h"

typedef float (*ACTIVATION)(float);

class ActivationHelper {
private:

    ActivationHelper() = delete;

    ActivationHelper(const ActivationHelper &activation_helper) = delete;

    virtual ~ActivationHelper() = default;

public:

    enum Type : long {
        SIGMOID = 0x00,
        RELU = 0x01,
    };

    static ACTIVATION get_activation(int id) {
        switch (id) {
            case SIGMOID:
                return sigmoid;
            case RELU:
                return relu;
            default:
                error_and_exit(
                        "ActivationHelper::get_activation() : failed to get activate function.");
                return nullptr;
        }
    }

    static ACTIVATION get_g_activation(int id) {
        switch (id) {
            case SIGMOID:
                return g_sigmoid;
            case RELU:
                return g_relu;
            default:
                error_and_exit(
                        "ActivationHelper::get_grad_activation() : failed to get activate function.");
                return nullptr;
        }
    }

};

#endif //CONVNETCPP_ACTIVATIONHELPER_H
