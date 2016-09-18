//
// Created by kanairen on 2016/06/15.
//

#include <map>

#include "config.h"
#include "MNIST.h"
#include "ShapeMap.h"
#include "SoftMaxLayer.h"
#include "ConvLayer.h"
#include "MaxPoolLayer.h"
#include "Model.h"
#include "optimizer.h"

#ifdef CONV_NET_CPP_DEBUG

namespace cnc {

    const float LEARNING_RATE = 0.01f;

    const unsigned int BATCH_SIZE = 25;

    const unsigned int MNIST_WIDTH = 28;
    const unsigned int MNIST_HEIGHT = 28;
    const unsigned int INPUT_SIZE = MNIST_WIDTH * MNIST_HEIGHT;

    const unsigned int N_ITERATION = 1000;
    const unsigned int N_CLASS = 10;

    // full connect layer
    const unsigned int N_HIDDEN_UNITS = 10000;

    float (*const LAYER_ACTIVATION)(float) = relu;

    float (*const LAYER_GRAD_ACTIVATION)(float) = g_relu;

    // grid layer
    const unsigned int C_IN = 1;
    const unsigned int C_OUT = 16;

    const unsigned int KERNEL_WIDTH = 2;
    const unsigned int KERNEL_HEIGHT = 2;

    const unsigned int STRIDE_X = 1;
    const unsigned int STRIDE_Y = 1;

    const unsigned int PADDING_X = 0;
    const unsigned int PADDING_Y = 0;

    float (*const CONV_ACTIVATION)(float) = iden;

    float (*const CONV_GRAD_ACTIVATION)(float) = g_iden;


};

void mnist_full_connect(char *argv[]) {

    string f_x_train = argv[2];
    string f_x_test = argv[3];
    string f_y_train = argv[4];
    string f_y_test = argv[5];

    // mnist
    MNIST mnist(f_x_train, f_x_test, f_y_train, f_y_test);

    Layer *layer_1 = new Layer(cnc::BATCH_SIZE,
                               cnc::INPUT_SIZE,
                               cnc::N_HIDDEN_UNITS,
                               cnc::LAYER_ACTIVATION,
                               cnc::LAYER_GRAD_ACTIVATION);

    Layer *layer_2 = new SoftMaxLayer(cnc::BATCH_SIZE,
                                      layer_1->get_n_out(),
                                      cnc::N_CLASS);

    vector<Layer *> v{layer_1, layer_2};

    // optimize
    optimize(mnist, v, cnc::LEARNING_RATE, cnc::BATCH_SIZE,
             cnc::N_ITERATION, cnc::N_CLASS);

    // release
    delete layer_1;
    delete layer_2;
}

void mnist_conv(char *argv[]) {

    string f_x_train = argv[2];
    string f_x_test = argv[3];
    string f_y_train = argv[4];
    string f_y_test = argv[5];

    // mnist
    MNIST mnist(f_x_train, f_x_test, f_y_train, f_y_test);

    Layer *layer_1 = new ConvLayer2d(cnc::BATCH_SIZE,
                                     cnc::MNIST_WIDTH, cnc::MNIST_HEIGHT,
                                     cnc::C_IN, cnc::C_OUT,
                                     cnc::KERNEL_WIDTH, cnc::KERNEL_HEIGHT,
                                     cnc::STRIDE_X, cnc::STRIDE_Y,
                                     cnc::PADDING_X, cnc::PADDING_Y,
                                     cnc::CONV_ACTIVATION,
                                     cnc::CONV_GRAD_ACTIVATION);

    Layer *layer_2 = new SoftMaxLayer(cnc::BATCH_SIZE,
                                      layer_1->get_n_out(),
                                      cnc::N_CLASS);

    vector<Layer *> v{layer_1, layer_2};

    std::cout << "conv:n_out:" << layer_1->get_n_out() << std::endl;

    // optimize
    optimize(mnist, v, cnc::LEARNING_RATE, cnc::BATCH_SIZE, cnc::N_ITERATION,
             cnc::N_CLASS);

    // release
    delete layer_1;
    delete layer_2;

}


void mnist_conv_pool(char *argv[]) {

    string f_x_train = argv[2];
    string f_x_test = argv[3];
    string f_y_train = argv[4];
    string f_y_test = argv[5];

    // mnist
    MNIST mnist(f_x_train, f_x_test, f_y_train, f_y_test);

    GridLayer2d *layer_1 = new ConvLayer2d(cnc::BATCH_SIZE,
                                           cnc::MNIST_WIDTH, cnc::MNIST_HEIGHT,
                                           cnc::C_IN, cnc::C_OUT,
                                           cnc::KERNEL_WIDTH,
                                           cnc::KERNEL_HEIGHT,
                                           cnc::STRIDE_X, cnc::STRIDE_Y,
                                           cnc::PADDING_X, cnc::PADDING_Y,
                                           cnc::CONV_ACTIVATION,
                                           cnc::CONV_GRAD_ACTIVATION);

    Layer *layer_2 = new MaxPoolLayer2d(cnc::BATCH_SIZE,
                                        layer_1->get_output_width(),
                                        layer_1->get_output_height(),
                                        cnc::C_OUT,
                                        cnc::KERNEL_WIDTH, cnc::KERNEL_HEIGHT,
                                        cnc::PADDING_X, cnc::PADDING_Y);

    Layer *layer_3 = new SoftMaxLayer(cnc::BATCH_SIZE,
                                      layer_2->get_n_out(),
                                      cnc::N_CLASS);

    vector<Layer *> v{layer_1, layer_2, layer_3};

    // optimize
    optimize(mnist, v, cnc::LEARNING_RATE, cnc::BATCH_SIZE, cnc::N_ITERATION,
             cnc::N_CLASS);

    // release
    delete layer_1;
    delete layer_2;
    delete layer_3;

}


void mnist_full_connect_eigen(char *argv[]) {

    string f_x_train = argv[2];
    string f_x_test = argv[3];
    string f_y_train = argv[4];
    string f_y_test = argv[5];

    // mnist
    MNIST mnist(f_x_train, f_x_test, f_y_train, f_y_test);

    Layer_ *layer_1 = new Layer_(cnc::BATCH_SIZE,
                                 cnc::INPUT_SIZE,
                                 cnc::N_HIDDEN_UNITS,
                                 cnc::LAYER_ACTIVATION,
                                 cnc::LAYER_GRAD_ACTIVATION);

    Layer_ *layer_2 = new SoftMaxLayer_(cnc::BATCH_SIZE,
                                        layer_1->get_n_out(),
                                        cnc::N_CLASS);

    vector<Layer_ *> v{layer_1, layer_2};

    // optimize
    optimize_(mnist, v, cnc::LEARNING_RATE, cnc::BATCH_SIZE,
              cnc::N_ITERATION, cnc::N_CLASS, argv[6], argv[7]);

    // release
    delete layer_1;
    delete layer_2;
}

void mnist_conv_eigen(char *argv[]) {

    string f_x_train = argv[2];
    string f_x_test = argv[3];
    string f_y_train = argv[4];
    string f_y_test = argv[5];

    // mnist
    MNIST mnist(f_x_train, f_x_test, f_y_train, f_y_test);

    Layer_ *layer_1 = new ConvLayer2d_(cnc::BATCH_SIZE,
                                       cnc::MNIST_WIDTH, cnc::MNIST_HEIGHT,
                                       cnc::C_IN, cnc::C_OUT,
                                       cnc::KERNEL_WIDTH, cnc::KERNEL_HEIGHT,
                                       cnc::STRIDE_X, cnc::STRIDE_Y,
                                       cnc::PADDING_X, cnc::PADDING_Y,
                                       cnc::CONV_ACTIVATION,
                                       cnc::CONV_GRAD_ACTIVATION);

    Layer_ *layer_2 = new SoftMaxLayer_(cnc::BATCH_SIZE,
                                        layer_1->get_n_out(),
                                        cnc::N_CLASS);

    vector<Layer_ *> v{layer_1, layer_2};

    std::cout << "conv:n_out:" << layer_1->get_n_out() << std::endl;

    // optimize
    optimize_(mnist, v, cnc::LEARNING_RATE, cnc::BATCH_SIZE, cnc::N_ITERATION,
              cnc::N_CLASS, argv[6], argv[7]);

    // release
    delete layer_1;
    delete layer_2;

}


void mnist_conv_pool_eigen(char *argv[]) {

    string f_x_train = argv[2];
    string f_x_test = argv[3];
    string f_y_train = argv[4];
    string f_y_test = argv[5];

    // mnist
    MNIST mnist(f_x_train, f_x_test, f_y_train, f_y_test);

    GridLayer2d_ *layer_1 = new ConvLayer2d_(cnc::BATCH_SIZE,
                                             cnc::MNIST_WIDTH,
                                             cnc::MNIST_HEIGHT,
                                             cnc::C_IN, cnc::C_OUT,
                                             cnc::KERNEL_WIDTH,
                                             cnc::KERNEL_HEIGHT,
                                             cnc::STRIDE_X, cnc::STRIDE_Y,
                                             cnc::PADDING_X, cnc::PADDING_Y,
                                             cnc::CONV_ACTIVATION,
                                             cnc::CONV_GRAD_ACTIVATION);

    Layer_ *layer_2 = new MaxPoolLayer2d_(cnc::BATCH_SIZE,
                                          layer_1->get_output_width(),
                                          layer_1->get_output_height(),
                                          cnc::C_OUT,
                                          cnc::KERNEL_WIDTH, cnc::KERNEL_HEIGHT,
                                          cnc::PADDING_X, cnc::PADDING_Y);

    Layer_ *layer_3 = new SoftMaxLayer_(cnc::BATCH_SIZE,
                                        layer_2->get_n_out(),
                                        cnc::N_CLASS);

    vector<Layer_ *> v{layer_1, layer_2, layer_3};

    // optimize
    optimize_(mnist, v, cnc::LEARNING_RATE, cnc::BATCH_SIZE, cnc::N_ITERATION,
              cnc::N_CLASS, argv[6], argv[7]);
    // release
    delete layer_1;
    delete layer_2;
    delete layer_3;

}

void shape_map_fc(char **argv) {

    ShapeMapSet shape_map_set(argv[2], argv[3]);

    // shuffle
    ShapeMapSet::shuffle(shape_map_set.x_train, shape_map_set.y_train);
    ShapeMapSet::shuffle(shape_map_set.x_test, shape_map_set.y_test);

    unsigned int batch_size = 1;
    unsigned int input_size = shape_map_set.data_size();
    unsigned int n_hidden_units = 100;
    unsigned int n_class = 2;
    unsigned int n_iter = 1000;

    float learning_rate = 0.001f;

    float (*const layer_activation)(float) = relu;

    float (*const layer_grad_activation)(float) = g_relu;

    Layer_ *layer_1 = new Layer_(batch_size,
                                 input_size,
                                 n_hidden_units,
                                 layer_activation,
                                 layer_grad_activation);

    Layer_ *layer_2 = new Layer_(batch_size,
                                 layer_1->get_n_out(),
                                 n_hidden_units,
                                 layer_activation,
                                 layer_grad_activation);

    Layer_ *layer_3 = new SoftMaxLayer_(batch_size,
                                        layer_1->get_n_out(),
                                        n_class);

    vector<Layer_ *> layers{layer_1, layer_2, layer_3};

    // optimize
    optimize_(shape_map_set, layers, learning_rate, batch_size, n_iter,
              n_class, argv[4], argv[5]);

    // release
    delete layer_1;
    delete layer_2;
    delete layer_3;

}


typedef void (*func_mnist)(char **);


// コマンドライン引数にmnistへのパスを渡す
int main(int argc, char *argv[]) {

    std::map<string, func_mnist> functions;

    functions["full_connect"] = mnist_full_connect;
    functions["conv"] = mnist_conv;
    functions["conv_pool"] = mnist_conv_pool;

    functions["full_connect_eigen"] = mnist_full_connect_eigen;
    functions["conv_eigen"] = mnist_conv_eigen;
    functions["conv_pool_eigen"] = mnist_conv_pool_eigen;

    functions["shape_map_fc"] = shape_map_fc;

    if (argc == 0) {
        std::cerr << "The number of command line arguments is incorrect." <<
        std::endl;
        exit(1);
    }

    string function_name = argv[1];

    if (functions.find(function_name) == functions.end()) {
        std::cerr << "function name is incorrect." << std::endl;
        exit(1);
    } else {
        functions[function_name](argv);
    }


}

#endif