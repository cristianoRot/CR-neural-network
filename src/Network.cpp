// network.cpp

#include "Network.hpp"
#include "TrainingLogger.hpp"
#include "ModelIO.hpp"
#include <cmath>
#include <string>

Network::Network(std::vector<Layer> layers_param, double learning_rate, InitType init_type, Loss loss_type)
    : layers(layers_param),
      learning_rate(learning_rate),
      loss_type(loss_type),
      accumulated_loss(0.0)
{
    if (layers.size() < 2)
    {
        throw std::invalid_argument("Error: Network must have at least 2 layers");
    }
    
    for (size_t i = 1; i < layers.size(); i++)
    {
        layers[i].connect_prev(layers[i - 1]);
    }

    init_weights(init_type);
}

// Init weights

void Network::init_weights(InitType init_type)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].init_weights(init_type);
    }
}

const Matrix& Network::get_output() const { return layers.back().getA(); }

void Network::train(Dataset& dataset, size_t epochs)
{
    dataset_size = dataset.size();
    TrainingLogger logger;

    for (size_t epoch = 0; epoch <= epochs; epoch++)
    {
        dataset.shuffle();

        for (size_t i = 0; i < dataset.size(); i++)
        {
            forward(dataset.get_input(i));

            size_t label = dataset.get_output(i);
            Matrix& pred = layers.back().getA();
            
            accumulate_loss(pred, label);
            compute_accuracy(pred, label);

            backprop(label);
            step(learning_rate);
        }

        accuracy = static_cast<double>(correct_predictions) / dataset_size;
        double avg_loss = accumulated_loss / dataset_size;
        
        logger.log_epoch(epoch, epochs, accuracy, avg_loss);
        
        lr_reduce_on_plateau();
        reset_epoch_metrics();
    }

    logger.log_completion();
}

void Network::forward(const Matrix& input)
{
    layers[0].set_prev_A(&input);

    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].forward(); 
    }
}

void Network::backprop(size_t label)
{
    loss_gradient(label);

    for (size_t i = layers.size(); i-- > 0; )
    {
        layers[i].backprop();
    }
}

void Network::loss_gradient(size_t label)
{
    const Matrix& prediction = layers.back().getA();

    switch (loss_type)
    {
        case Loss::CROSS_ENTROPY:
        {
            Matrix dZ = prediction;
            double v = dZ.get(label, 0);
            dZ.set(label, 0, v - 1.0);
            layers.back().set_dZ(dZ);

            break;
        }
        case Loss::MSE:
        {
            Matrix target(prediction.rows(), 1);
            target.fill(0.0);
            target.set(label, 0, 1.0);
            
            Matrix dZ = prediction - target;

            for (size_t i = 0; i < dZ.rows(); i++)
            {
                double val = dZ.get(i, 0);
                dZ.set(i, 0, 2.0 * val);
            }

            layers.back().set_dZ(dZ);

            break;
        }
    }
}

void Network::accumulate_loss(const Matrix& prediction, size_t label)
{
    switch (loss_type)
    {
        case Loss::CROSS_ENTROPY:
        {
            double pred_prob = prediction.get(label, 0);
            if (pred_prob < 1e-10) pred_prob = 1e-10; // Avoid log(0)
            accumulated_loss += -std::log(pred_prob);
            break;
        }
        case Loss::MSE:
        {
            // Create target vector (one-hot encoding)
            Matrix target(prediction.rows(), 1);
            target.fill(0.0);
            target.set(label, 0, 1.0);
            
            // MSE: sum of squared differences
            Matrix diff = prediction - target;
            double mse = 0.0;
            for (size_t i = 0; i < diff.rows(); i++)
            {
                double val = diff.get(i, 0);
                mse += val * val;
            }
            accumulated_loss += mse;
            break;
        }
    }
}

void Network::step(double learning_rate)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].step(learning_rate, 0.9);
    }
}

void Network::lr_reduce_on_plateau()
{
    if (accuracy > best_accuracy + min_delta)
    {
        best_accuracy = accuracy;
        patience_counter = 0;
        
        ModelIO::save_model(*this, "checkpoints/model.crnn");
        
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
}

void Network::reset_epoch_metrics()
{
    correct_predictions = 0;
    accumulated_loss = 0.0;
}

void Network::print_accuracy()
{
    accuracy = static_cast<double>(correct_predictions) / dataset_size;
    
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

void Network::load(const std::string& filepath)
{
    ModelIO::load_model(*this, filepath);
}

void Network::save(const std::string& filepath)
{
    ModelIO::save_model(*this, filepath);
}