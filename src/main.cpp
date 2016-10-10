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

#include "util/StringUtil.h"

#include "helper/ActivationHelper.h"
#include "helper/DataSetHelper.h"

#include "tinyxml/tinyxml2.h"

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLNode;

#ifdef CONV_NET_CPP_DEBUG


namespace xmlkey {
    constexpr char ROOT[] = "root";

    constexpr char N_ITER[] = "n_iteration";
    constexpr char LEARNING_RATE[] = "learning_rate";

    constexpr char DATA_SET[] = "data_set";
    constexpr char DATA_SET_ID[] = "id";
    constexpr char DATA_SET_IS_SHUFFLED[] = "is_shuffled";
    constexpr char DATA_SET_BATCH_SIZE[] = "batch_size";

    constexpr char NETS[] = "nets";

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

Layer_ *new_fc(XMLElement *xml_layer, int n_data, int n_in) {

    if (is_equal(xml_layer->Name(), xmlkey::FULL_CONNECT)) {
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
    bool is_weight_rand_init_enabled = atob(
            map[xmlkey::IS_WEIGHT_RAND_INIT_ENABLED].c_str());
    float weight_constant_value = (float) std::atof(
            map[xmlkey::WEIGHT_CONSTANT_VALUE].c_str());

    // dropout setting
    bool is_dropout_enabled = atob(map[xmlkey::IS_DROPOUT_ENABLED].c_str());
    float dropout_rate = (float) std::atof(map[xmlkey::DROPOUT_RATE].c_str());

    return new Layer_(n_data, n_in, n_hidden, act, g_act,
                      is_weight_rand_init_enabled, weight_constant_value,
                      is_dropout_enabled, dropout_rate);

}

SoftMaxLayer_ *new_softmax(XMLElement *xml_layer, int n_data, int n_in,
                           int n_class) {

    if (xml_layer->Name() == xmlkey::SOFTMAX) {
        error_and_exit(
                "new_softmax_layer(): a xml element in arguments is incorrect.");
    }

    std::map<string, string> &&map = params_softmax(xml_layer);

    // weight initialization setting
    bool is_weight_rand_init_enabled = atob(
            map[xmlkey::IS_WEIGHT_RAND_INIT_ENABLED].c_str());
    float weight_constant_value = (float) std::atof(
            map[xmlkey::WEIGHT_CONSTANT_VALUE].c_str());

    // dropout setting
    bool is_dropout_enabled = atob(map[xmlkey::IS_DROPOUT_ENABLED].c_str());
    float dropout_rate = (float) std::atof(map[xmlkey::DROPOUT_RATE].c_str());

    return new SoftMaxLayer_(n_data, n_in, n_class, is_weight_rand_init_enabled,
                             weight_constant_value, is_dropout_enabled,
                             dropout_rate);

}

int main(int argc, char *argv[]) {


    /*
     * Check Command Line Arguments
     */

    if (argc != 6) {
        string message = "\n  The number of arguments is too ";
        if (argc > 6) {
            message += "much! \n\n";
        } else {
            message += "short! \n\n";
        }
        message += " *.out [config xml path] "
                "[training data path] [test data path] "
                "[training log path] [test log path]";
        error_and_exit(message);
    }


    /*
     * XML Parse
     */

    XMLDocument xml;
    xml.LoadFile(argv[1]);

    // xml:root
    XMLElement *xml_root = xml.FirstChildElement(xmlkey::ROOT);

    // xml:n_iteration
    XMLElement *xml_n_iteration = xml_root->FirstChildElement(xmlkey::N_ITER);
    // xml:learning_rate
    XMLElement *xml_lr = xml_root->FirstChildElement(xmlkey::LEARNING_RATE);

    // xml:data_set
    XMLElement *xml_ds = xml_root->FirstChildElement(xmlkey::DATA_SET);
    XMLElement *xml_ds_id = xml_ds->FirstChildElement(xmlkey::DATA_SET_ID);
    XMLElement *xml_ds_is_shuffled = xml_ds->FirstChildElement(
            xmlkey::DATA_SET_IS_SHUFFLED);
    XMLElement *xml_batch_size = xml_ds->FirstChildElement(
            xmlkey::DATA_SET_BATCH_SIZE);

    // xml:layer_params
    std::vector<std::pair<string, string>> layer_params;
    XMLElement *xml_nets = xml_root->FirstChildElement(xmlkey::NETS);


    /*
     * Getting Start
     */

    // DataSet
    std::unique_ptr<DataSet<float, int>> data_set(DataSetHelper::get_dataset(
            std::atoi(xml_ds_id->GetText()), argv));
    // shuffle
    ShapeMapSet::shuffle(data_set->x_train, data_set->y_train);
    ShapeMapSet::shuffle(data_set->x_test, data_set->y_test);

    // Parameters for learning
    int n_class = data_set->get_n_cls(); // N-class
    int n_iter = std::atoi(xml_n_iteration->GetText()); // N-Iteration
    int batch_size = std::atoi(xml_batch_size->GetText());
    float learning_rate = (float) std::atof(xml_lr->GetText());

    // Layers
    vector<Layer_ *> layers;
    int n_in = data_set->data_size();
    XMLNode *node = xml_nets->FirstChild();
    while (node != nullptr) {

        XMLElement *net_elem = node->ToElement();

        if (is_equal(net_elem->Name(), xmlkey::FULL_CONNECT)) {
            layers.push_back(new_fc(net_elem, batch_size, n_in));
        } else if (is_equal(net_elem->Name(), xmlkey::SOFTMAX)) {
            layers.push_back(new_softmax(net_elem, batch_size, n_in, n_class));
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