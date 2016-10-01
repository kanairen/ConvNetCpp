//
// Created by kanairen on 2016/06/15.
//

#include <map>
#include <memory>

#include "config.h"
#include "Data.h"
#include "MNIST.h"
#include "ShapeMap.h"
#include "layer/SoftMaxLayer.h"
#include "layer/ConvLayer.h"
#include "layer/MaxPoolLayer.h"
#include "Model.h"
#include "Optimizer.h"

#include "tinyxml/tinyxml2.h"

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLNode;

#ifdef CONV_NET_CPP_DEBUG


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

namespace xmlkey {
    constexpr char FULL_CONNECT[] = "full_connect";
    constexpr char SOFTMAX[] = "softmax";

    constexpr char N_HIDDEN[] = "n_hidden";
    constexpr char ACTIVATION_ID[] = "activation_id";
    constexpr char IS_WEIGHT_RAND_INIT_ENABLED[] = "is_weight_rand_init_enabled";
    constexpr char WEIGHT_CONSTANT_VALUE[] = "weight_constant_value";
    constexpr char IS_DROPOUT_ENABLED[] = "is_dropout_enabled";
    constexpr char DROPOUT_RATE[] = "dropout_rate";
}

std::map<string, string> params_common(XMLElement *elem) {

    return {
            {xmlkey::IS_WEIGHT_RAND_INIT_ENABLED, elem->FirstChildElement(
                    xmlkey::IS_WEIGHT_RAND_INIT_ENABLED)->GetText()},
            {xmlkey::WEIGHT_CONSTANT_VALUE,       elem->FirstChildElement(
                    xmlkey::WEIGHT_CONSTANT_VALUE)->GetText()},
            {xmlkey::IS_DROPOUT_ENABLED,          elem->FirstChildElement(
                    xmlkey::IS_DROPOUT_ENABLED)->GetText()},
            {xmlkey::DROPOUT_RATE,                elem->FirstChildElement(
                    xmlkey::DROPOUT_RATE)->GetText()},
    };

}

std::map<string, string> params_layer(XMLElement *elem) {

    std::map<string, string> &&map = params_common(elem);
    map[xmlkey::N_HIDDEN] = elem->FirstChildElement(
            xmlkey::N_HIDDEN)->GetText();
    map[xmlkey::ACTIVATION_ID] = elem->FirstChildElement(
            xmlkey::ACTIVATION_ID)->GetText();

    return map;

};

std::map<string, string> params_softmax(XMLElement *elem) {
    return params_common(elem);
};

Layer_ *new_layer(XMLElement *xml_layer, int n_data, int n_in) {

    if (strncmp(xml_layer->Name(), xmlkey::FULL_CONNECT, 2) != 0) {
        error_and_exit("new_layer(): a xml element in arguments is incorrect.");
    }

    std::map<string, string> &&map = params_layer(xml_layer);

    // n-hidden
    int n_hidden = atoi(map[xmlkey::N_HIDDEN].c_str());
    // activation
    ACTIVATION act = ActivationHelper::get_activation(
            atoi(map[xmlkey::ACTIVATION_ID].c_str()));
    // grad-activation
    ACTIVATION g_act = ActivationHelper::get_g_activation(
            atoi(map[xmlkey::ACTIVATION_ID].c_str()));

    // weight initialization setting
    bool is_weight_rand_init_enabled = std::strncmp(
            map[xmlkey::IS_WEIGHT_RAND_INIT_ENABLED].c_str(), "true", 4) == 0;
    float weight_constant_value = std::atof(
            map[xmlkey::WEIGHT_CONSTANT_VALUE].c_str());

    // dropout setting
    bool is_dropout_enabled =
            std::strncmp(map[xmlkey::IS_DROPOUT_ENABLED].c_str(), "true", 4) ==
            0;
    float dropout_rate = std::atof(map[xmlkey::DROPOUT_RATE].c_str());

    return new Layer_(n_data, n_in, n_hidden, act, g_act,
                      is_weight_rand_init_enabled, weight_constant_value,
                      is_dropout_enabled, dropout_rate);

}

SoftMaxLayer_ *new_softmax_layer(XMLElement *xml_layer, int n_data, int n_in,
                                 int n_class) {

    if (strncmp(xml_layer->Name(), xmlkey::SOFTMAX, 2) != 0) {
        error_and_exit(
                "new_softmax_layer(): a xml element in arguments is incorrect.");
    }

    std::map<string, string> &&map = params_softmax(xml_layer);

    // weight initialization setting
    bool is_weight_rand_init_enabled = std::strncmp(
            map[xmlkey::IS_WEIGHT_RAND_INIT_ENABLED].c_str(), "true", 4) == 0;
    float weight_constant_value = std::atof(
            map[xmlkey::WEIGHT_CONSTANT_VALUE].c_str());

    // dropout setting
    bool is_dropout_enabled =
            std::strncmp(map[xmlkey::IS_DROPOUT_ENABLED].c_str(), "true", 4) ==
            0;
    float dropout_rate = std::atof(map[xmlkey::DROPOUT_RATE].c_str());

    return new SoftMaxLayer_(n_data, n_in, n_class, is_weight_rand_init_enabled,
                             weight_constant_value, is_dropout_enabled,
                             dropout_rate);

}

// コマンドライン引数にmnistへのパスを渡す
int main(int argc, char *argv[]) {

    /*
     * XML Parse
     */

    XMLDocument xml;
    xml.LoadFile(argv[1]);

    // xml:root
    XMLElement *xml_root = xml.FirstChildElement("root");

    // xml:n_iteration
    XMLElement *xml_n_iteration = xml_root->FirstChildElement("n_iteration");
    // xml:learning_rate
    XMLElement *xml_lr = xml_root->FirstChildElement("learning_rate");

    // xml:data_set
    XMLElement *xml_data_set = xml_root->FirstChildElement("data_set");
    XMLElement *xml_data_set_id = xml_data_set->FirstChildElement("id");
    XMLElement *xml_data_set_is_shuffled = xml_data_set->FirstChildElement(
            "is_shuffled");
    XMLElement *xml_batch_size = xml_data_set->FirstChildElement("batch_size");

    // xml:layer_params
    std::vector<std::pair<string, string>> layer_params;
    XMLElement *xml_nets = xml_root->FirstChildElement("nets");


    /*
     * Getting Start
     */

    // DataSet
    std::unique_ptr<DataSet<float, int>> data_set(DataSetHelper::get_dataset(
            std::atoi(xml_data_set_id->GetText()), argv));
    // shuffle
    ShapeMapSet::shuffle(data_set->x_train, data_set->y_train);
    ShapeMapSet::shuffle(data_set->x_test, data_set->y_test);

    // Parameters for learning
    int n_class = data_set->get_n_cls(); // N-class
    int n_iter = std::atoi(xml_n_iteration->GetText()); // N-Iteration
    int batch_size = std::atoi(xml_batch_size->GetText());
    float learning_rate = (float) std::atof(xml_lr->GetText());

    // Layers
    vector<Layer_*> layers;
    int n_in = data_set->data_size();
    XMLNode *node = xml_nets->FirstChild();
    while (node != nullptr) {
        XMLElement *xml_nets_elem = node->ToElement();
        if (std::strcmp(xml_nets_elem->Name(), xmlkey::FULL_CONNECT) == 0) {
            layers.push_back(new_layer(xml_nets_elem, batch_size, n_in));
        } else if (std::strcmp(xml_nets_elem->Name(), xmlkey::SOFTMAX) == 0) {
            layers.push_back(new_softmax_layer(xml_nets_elem, batch_size, n_in,
                                               n_class));
        } else {
            error_and_exit("failed to set layer.");
        }
        node = node->NextSibling();
        n_in = layers.back()->get_n_out();
    }

    // optimize
    optimize_(*data_set, layers, learning_rate, batch_size, n_iter,
              n_class, argv[argc - 2], argv[argc - 1]);


}

#endif