//
// Created by kanairen on 2016/06/15.
//

#include "config.h"
#include "MNIST.h"
#include "Layer.h"
#include "Model.h"
#include "optimizer.h"
#include "activation.h"

#ifdef CONV_NET_CPP_DEBUG


// コマンドライン引数にmnistへのパスを渡す
int main(int argc, char *argv[]) {

    float learning_rate = 0.01f;
    unsigned int batch_size = 10;
    unsigned int input_size = 28 * 28;
    unsigned int n_iter = 1000;
    unsigned int n_class = 10;

    // mnist
    MNIST mnist(argv[1], argv[2], argv[3], argv[4]);
    vector<Layer> v{Layer(batch_size, mnist.xv_size(), n_class, iden, g_iden)};

    // optimize
    optimize(mnist, v, learning_rate,  batch_size, n_iter, n_class);

}

#endif