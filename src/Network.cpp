// network.cpp

#include "Network.hpp"
#include "Dataset.hpp"
#include "TrainingLogger.hpp"
#include "ModelIO.hpp"
#include "MetricsHandler.hpp"
#include <cstddef>
#include <string>

Network::Network(std::vector<Layer> layers_param, double learning_rate, InitType init_type, Loss loss_type)
    : layers(layers_param),
      loss_type(loss_type),
      optimizer(learning_rate)
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

Metrics Network::eval(Dataset& dataset)
{
    set_mode(Mode::EVAL);
    class_weights = dataset.get_class_weight();
    
    return do_pass(dataset, 1);
}

void Network::train(Dataset& dataset, size_t epochs, size_t batch_size)
{
    dataset_size = dataset.size();
    class_weights = dataset.get_class_weight();


    TrainingLogger logger;

    for (size_t epoch = 0; epoch <= epochs; epoch++)
    {
        set_mode(Mode::TRAIN);
        dataset.shuffle();

        Metrics train_metrics = do_pass(dataset, batch_size);
        double train_accuracy = train_metrics.accuracy;
        double train_loss = train_metrics.loss;
        
        Metrics eval_metrics = eval(dataset);
        double eval_accuracy = eval_metrics.accuracy;
        double eval_loss = eval_metrics.loss;
        
        logger.log_epoch(epoch, epochs, train_accuracy, train_loss, eval_accuracy, eval_loss);
        optimizer.lr_reduce_on_plateau(eval_accuracy, *this);
    }

    logger.log_completion();
}

Metrics Network::do_pass(Dataset& dataset, size_t batch_size)
{
    size_t correct_predictions = 0;
    double accumulated_loss = 0.0;

    for (size_t i = 0; i < dataset.size(); i += batch_size)
    {
        Matrix input = dataset.get_input(i, batch_size);
        std::vector<size_t> labels = dataset.get_output(i, batch_size);

        forward(input);

        Matrix& pred = layers.back().getA();
        
        accumulated_loss = MetricsHandler::accumulate_loss(pred, labels, accumulated_loss, loss_type, class_weights);
        correct_predictions += MetricsHandler::count_correct_predictions(pred, labels);

        if (mode == Mode::EVAL) continue;
        
        backprop(labels);
        optimizer.step(layers);
    }

    double acc = static_cast<double>(correct_predictions) / dataset.size();
    double loss = accumulated_loss / dataset.size();

    return {acc, loss};
}

void Network::forward(const Matrix& input_batch)
{
    layers[0].set_prev_A(&input_batch);

    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].set_mode(mode);
        layers[i].forward(); 
    }
}

void Network::backprop(const std::vector<size_t>& labels)
{
    Layer& last_layer = layers.back();
    
    // Compute loss gradient internally updates last_layer's dZ and/or dA
    MetricsHandler::compute_loss_gradient(last_layer, labels, loss_type, class_weights);

    for (size_t i = layers.size(); i-- > 0; )
    {
        layers[i].backprop();
    }
}

void Network::load(const std::string& filepath)
{
    ModelIO::load_model(*this, filepath);
}

void Network::save(const std::string& filepath)
{
    ModelIO::save_model(*this, filepath);
}