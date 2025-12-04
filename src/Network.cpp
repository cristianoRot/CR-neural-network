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

    Layer& last_layer = layers.back();

    if (loss_type == Loss::CROSS_ENTROPY && last_layer.get_activation() != Activation::SOFTMAX)
    {
        throw std::invalid_argument(
            "Error: Cross-entropy loss requires SOFTMAX activation in the last layer"
        );
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

void Network::train(Dataset& dataset, size_t epochs, size_t batch_size)
{
    dataset_size = dataset.size();
    class_weights = dataset.get_class_weight();

    TrainingLogger logger;

    for (size_t epoch = 0; epoch <= epochs; epoch++)
    {
        dataset.shuffle();

        for (size_t i = 0; i < dataset.size(); i += batch_size)
        {
            Matrix input_batch = dataset.get_input(i, batch_size);
            std::vector<size_t> labels = dataset.get_output(i, batch_size);

            forward(input_batch);

            Matrix& pred = layers.back().getA();
            
            accumulate_loss(pred, labels);
            compute_accuracy(pred, labels);

            backprop(labels);
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

void Network::forward(const Matrix& input_batch)
{
    layers[0].set_prev_A(&input_batch);

    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].forward(); 
    }
}

void Network::backprop(const std::vector<size_t>& labels)
{
    loss_gradient(labels);

    for (size_t i = layers.size(); i-- > 0; )
    {
        layers[i].backprop();
    }
}

void Network::step(double learning_rate)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].step(learning_rate, 0.9);
    }
}

void Network::loss_gradient(const std::vector<size_t>& labels)
{
    const Matrix& prediction = layers.back().getA();
    size_t batch_size = prediction.cols();
    size_t num_classes = prediction.rows();

    if (labels.size() != batch_size)
        throw std::invalid_argument("Error: Labels must have the same size as the batch size");

    switch (loss_type)
    {
        case Loss::CROSS_ENTROPY:
        {
            Matrix dZ = prediction;

            for (size_t i = 0; i < batch_size; i++)
            {
                size_t label = labels[i];
                double weight = class_weights[label];

                double v = dZ.get(label, i);
                dZ.set(label, i, v - 1.0);
                dZ.multiply_col(i, weight);
            }

            dZ /= batch_size;
            layers.back().set_dZ(dZ);

            break;
        }
        case Loss::MSE:
        {
            Matrix dZ(num_classes, batch_size);

            for (size_t i = 0; i < batch_size; ++i)
            {
                size_t label = labels[i];
                double weight = class_weights[label];
                
                for (size_t cls = 0; cls < num_classes; ++cls)
                {
                    double target = (cls == label) ? 1.0 : 0.0;
                    double pred   = prediction.get(cls, i);
                    double diff   = pred - target;
                    dZ.set(cls, i, 2.0 * diff * weight);
                }
            }

            dZ /= batch_size;
            layers.back().set_dZ(dZ);

            break;
        }
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

void Network::compute_accuracy(const Matrix& prediction, const std::vector<size_t>& labels)
{
    const size_t B = prediction.cols();

    for (size_t i = 0; i < B; i++)
    {
        size_t label = labels[i];
        size_t argmax = prediction.argmax_col(i);

        if (argmax == label) correct_predictions++;
    }
}

void Network::accumulate_loss(const Matrix& prediction, const std::vector<size_t>& labels)
{
    const size_t B = prediction.cols();
    double batch_loss = 0.0;

    switch (loss_type)
    {
        case Loss::CROSS_ENTROPY:
        {
            for (size_t i = 0; i < B; i++)
            {
                size_t label = labels[i];
                double weight = class_weights[label];

                double pred_prob = prediction.get(label, i);
                if (pred_prob < 1e-10) pred_prob = 1e-10; // Avoid log(0)
                batch_loss -= weight * std::log(pred_prob);
            }
            break;
        }
        case Loss::MSE:
        {
            const size_t C = prediction.rows();

            for (size_t i = 0; i < B; ++i)
            {
                size_t label = labels[i];
                double weight = class_weights[label];

                double mse_sample = 0.0;

                for (size_t cls = 0; cls < C; ++cls)
                {
                    double target = (cls == label) ? 1.0 : 0.0;
                    double pred   = prediction.get(cls, i);
                    double diff   = pred - target;
                    mse_sample   += weight * diff * diff;
                }

                batch_loss += mse_sample;
            }
            break;
        }
    }

    accumulated_loss += batch_loss;
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

void Network::load(const std::string& filepath)
{
    ModelIO::load_model(*this, filepath);
}

void Network::save(const std::string& filepath)
{
    ModelIO::save_model(*this, filepath);
}