// network.cpp

#include "Network.hpp"

Network::Network(std::vector<size_t> layer_sizes, InitType init_type, double learning_rate)
{
    if (layer_sizes.size() < 2)
    {
        throw std::invalid_argument("Error: Network must have at least 2 layers");
    }

    this->learning_rate = learning_rate;

    input_layer = Matrix(layer_sizes[0], 1);

    const Matrix* prev_A = &input_layer;
    Matrix* prev_dA = nullptr;

    layers.reserve(layer_sizes.size() - 2);
    
    for (size_t i = 1; i < layer_sizes.size() - 1; i++)
    {
        layers.push_back(HiddenLayer(layer_sizes[i - 1], 
                                    layer_sizes[i],
                                    prev_A,
                                    prev_dA));

        HiddenLayer& last_layer = layers.back();
        
        prev_A = &last_layer.getA();
        prev_dA = &last_layer.get_dA();
    }

    size_t vector_size = layer_sizes.size();
    output_layer = std::make_unique<OutputLayer>(
        layer_sizes[vector_size - 2],
        layer_sizes[vector_size - 1],
        prev_A,
        prev_dA
    );

    init_weights(init_type);
}

// Init weights

void Network::init_weights(InitType init_type)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].init_weights(init_type);
    }

    output_layer->init_weights(init_type);
}

const Matrix& Network::get_output() const { return output_layer->getA(); }

void Network::train(Dataset& dataset, size_t epochs)
{
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        dataset.shuffle();

        std::cout << "Epoch " << epoch << "..." << std::endl;

        for (size_t i = 0; i < dataset.size(); i++)
        {
            forward(dataset.get_input(i));

            size_t label = dataset.get_output(i);

            compute_accuracy(output_layer->getA(), label);

            backprop(label);
            step(learning_rate);
        }

        print_accuracy();
        
        lr_reduce_on_plateau();
        
        reset_accuracy();
    }
}

void Network::forward(const Matrix& input)
{
    input_layer = input; // Set input layer

    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].forward(); 
    }

    output_layer->forward();
}

void Network::backprop(size_t label)
{
    loss_gradient(label);
    output_layer->backprop();

    for (size_t i = layers.size(); i-- > 0; )
    {
        layers[i].backprop();
    }
}

void Network::loss_gradient(size_t label)
{
    Matrix dZ = output_layer->getA();
    double v = dZ.get(label, 0);

    dZ.set(label, 0, v - 1.0);
    output_layer->set_dZ(dZ);
}

void Network::step(double learning_rate)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].step(learning_rate, 0.9);
    }

    output_layer->step(learning_rate, 0.9);
}

void Network::lr_reduce_on_plateau()
{
    if (accuracy > best_accuracy + min_delta)
    {
        best_accuracy = accuracy;
        patience_counter = 0;
        return;
    }

    patience_counter++;
    
    if (patience_counter >= patience)
    {
        double new_lr = learning_rate * factor;
        
        if (new_lr >= min_lr)
        {
            learning_rate = new_lr;            
            best_accuracy = accuracy;
        }
        
        patience_counter = 0;
    }
}

void Network::compute_accuracy(const Matrix& prediction, size_t label)
{
    size_t argmax = Network::argmax(prediction);

    if (argmax == label) correct_predictions++;
    
    total_predictions++;
}

void Network::reset_accuracy()
{
    correct_predictions = 0;
    total_predictions = 0;
}

void Network::print_accuracy()
{
    accuracy = static_cast<double>(correct_predictions) / total_predictions;
    
    std::cout << "Accuracy: " << accuracy << std::endl;
}

size_t Network::argmax(const Matrix& prediction)
{
    size_t max_idx = 0;
    double max_val = prediction.get(0, 0);

    for (size_t i = 1; i < prediction.rows(); i++)
    {
        if (prediction.get(i, 0) > max_val)
        {
            max_idx = i;
            max_val = prediction.get(i, 0);
        }
    }
    return max_idx;
}