// network.hpp

#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Dataset.hpp"
#include "InitType.hpp"
#include <vector>
#include <tuple>
#include <memory>


class Network 
{
    private:
        Matrix input_layer;
        std::vector<HiddenLayer> layers;
        std::unique_ptr<OutputLayer> output_layer;

        double learning_rate;
        double accuracy = 0.0;

        size_t correct_predictions = 0;
        size_t total_predictions = 0;

    public:
        Network(std::vector<size_t> layer_sizes, InitType init_type, double learning_rate);

        void init_weights(InitType init_type);
        const Matrix& get_output() const;

        void train(Dataset& dataset, size_t epochs);
        void forward(const Matrix& input);
        void backprop(size_t label);
        void step(double learning_rate);

        double lr_step_decay(int epoch, double lr0, int drop_every, double gamma);

        void loss_gradient(size_t label);

        void compute_accuracy(const Matrix& prediction, size_t label);
        void reset_accuracy();
        void print_accuracy();

        size_t argmax(const Matrix& prediction);
};
