//
// Created by kanairen on 2016/06/15.
//

#include "config.h"
#include "MNIST.h"
#include "Layer.h"
#include "ConvLayer.h"
#include "Model.h"
#include "optimizer.h"
#include "activation.h"

#ifdef CONV_NET_CPP_DEBUG


// コマンドライン引数にmnistへのパスを渡す
int main(int argc, char *argv[]) {

    float learning_rate = 0.01f;
    unsigned int batch_size = 50;
    unsigned int input_size = 28 * 28;
    unsigned int n_iter = 1000;
    unsigned int n_class = 10;

    // mnist
    MNIST mnist(argv[1], argv[2], argv[3], argv[4]);

    AbstractLayer *layer_1 = new ConvLayer2d(batch_size, 28, 28, 1, 16, 2, 2, 1,
                                             iden, g_iden);
    AbstractLayer *layer_2 = new Layer(batch_size, layer_1->get_n_out(),
                                       n_class, iden, g_iden);
    vector<AbstractLayer *> v{layer_1, layer_2};
//    AbstractLayer* layer_1 = new Layer(batch_size, mnist.xv_size(), 10, iden, g_iden);
//    AbstractLayer* layer_2 = new Layer(batch_size, 10, n_class, iden, g_iden);
//    vector<AbstractLayer*> v{layer_1,layer_2};

    // optimize
    optimize(mnist, v, learning_rate, batch_size, n_iter, n_class);

    // release
    delete (layer_1);

}

#endif