// network.hpp

#pragma once
#include "Functions.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Dataset.hpp"
#include <vector>
#include <tuple>
#include <memory>

class Network 
{
    private:
        std::vector<Layer> layers;

        double learning_rate;
        double accumulated_loss = 0.0;
        double accuracy = 0.0;
        
        // Learning rate reduction on plateau
        double best_accuracy = 0.0;
        size_t patience_counter = 0;
        size_t patience = 20;
        double factor = 0.7;
        double min_lr = 1e-6;
        double min_delta = 0.001;

        size_t correct_predictions = 0;
        size_t dataset_size = 0;

    public:
        Network(std::vector<Layer> layers, InitType init_type, double learning_rate);

        void init_weights(InitType init_type);
        const Matrix& get_output() const;

        void train(Dataset& dataset, size_t epochs);
        void forward(const Matrix& input);
        void backprop(size_t label);
        void step(double learning_rate);

        void lr_reduce_on_plateau();

        void loss_gradient(size_t label);
        void accumulate_loss(const Matrix& prediction, size_t label);

        void compute_accuracy(const Matrix& prediction, size_t label);
        void reset_epoch_metrics();
        void print_accuracy();

        size_t argmax(const Matrix& prediction);
};
