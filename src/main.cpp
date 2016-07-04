//
// Created by kanairen on 2016/06/15.
//

#include "config.h"
#include "MNIST.h"
#include "ConvLayer.h"
#include "Model.h"
#include "optimizer.h"
#include "activation.h"

#ifdef CONV_NET_CPP_DEBUG

void mnist_full_connelct(char *argv[],
                         unsigned int batch_size,
                         unsigned int input_size,
                         unsigned int n_class,
                         unsigned int n_iter,
                         float learning_rate) {
    // mnist
    MNIST mnist(argv[1], argv[2], argv[3], argv[4]);

    Layer *layer_1 = new Layer(batch_size, input_size, n_class,
                               iden, g_iden);
    vector<Layer *> v{layer_1};

    // optimize
    optimize(mnist, v, learning_rate, batch_size, n_iter, n_class);

    // release
    delete (layer_1);
}

void mnist_conv(char *argv[], unsigned int batch_size,
                unsigned int input_width, unsigned int input_height,
                unsigned int c_in, unsigned int c_out,
                unsigned int kw, unsigned int kh, unsigned int stride,
                unsigned int n_class, unsigned int n_iter,
                float learning_rate) {
    // mnist
    MNIST mnist(argv[1], argv[2], argv[3], argv[4]);

    Layer *layer_1 = new ConvLayer2d(batch_size,
                                     input_width, input_height,
                                     c_in, c_out, kw, kh, stride,
                                     iden, g_iden);

    Layer *layer_2 = new Layer(batch_size, layer_1->get_n_out(),
                               n_class, iden, g_iden);

    vector<Layer *> v{layer_1, layer_2};

    std::cout << "conv:n_out:" << layer_1->get_n_out() << std::endl;

    // optimize
    optimize(mnist, v, learning_rate, batch_size, n_iter, n_class);

    // release
    delete (layer_1);
    delete (layer_2);

}


// コマンドライン引数にmnistへのパスを渡す
int main(int argc, char *argv[]) {

    const float LEARNING_RATE = 0.01f;

    const unsigned int BATCH_SIZE = 25;

    const unsigned int WIDTH = 28;
    const unsigned int HEIGHT = 28;
    const unsigned int INPUT_SIZE = WIDTH * HEIGHT;

    const unsigned int C_IN = 1;
    const unsigned int C_OUT = 16;

    const unsigned int KERNEL_WIDTH = 2;
    const unsigned int KERNEL_HEIGHT = 2;

    const unsigned int STRIDE = 1;

    const unsigned int N_ITERATION = 1000;
    const unsigned int N_CLASS = 10;

    mnist_conv(argv, BATCH_SIZE, WIDTH, HEIGHT, C_IN, C_OUT, KERNEL_WIDTH,
               KERNEL_HEIGHT, STRIDE, N_CLASS, N_ITERATION, LEARNING_RATE);
//    mnist_full_connelct(argv, BATCH_SIZE, INPUT_SIZE, N_CLASS, N_ITERATION,
//                        LEARNING_RATE);


}

#endif