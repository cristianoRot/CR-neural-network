// network.hpp

#pragma once
#include "Functions.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Dataset.hpp"
#include <vector>
#include <tuple>
#include <memory>
#include <string>

class Network 
{
    private:
        std::vector<Layer> layers;
        std::vector<double> class_weights;
        Loss loss_type;

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
        Network(std::vector<Layer> layers, double learning_rate, InitType init_type, Loss loss_type = Loss::CROSS_ENTROPY);

        void init_weights(InitType init_type);
        void load(const std::string& filepath);
        void save(const std::string& filepath);
        const Matrix& get_output() const;

        void train(Dataset& dataset, size_t epochs, size_t batch_size);
        void forward(const Matrix& input);
        void backprop(const std::vector<size_t>& labels);
        void step(double learning_rate);

        void loss_gradient(const std::vector<size_t>& labels);
        void lr_reduce_on_plateau();
        void compute_accuracy(const Matrix& prediction, const std::vector<size_t>& labels);
        void accumulate_loss(const Matrix& prediction, const std::vector<size_t>& labels);

        void reset_epoch_metrics();
        void print_accuracy();
        
        // Model I/O getters
        std::vector<Layer>& get_layers() { return layers; }
        const std::vector<Layer>& get_layers() const { return layers; }
        Loss get_loss_type() const { return loss_type; }
        double get_learning_rate() const { return learning_rate; }
        double get_best_accuracy() const { return best_accuracy; }
        size_t get_patience() const { return patience; }
        double get_factor() const { return factor; }
        double get_min_lr() const { return min_lr; }
        double get_min_delta() const { return min_delta; }
        
        // Model I/O setters
        void set_learning_rate(double lr) { learning_rate = lr; }
        void set_best_accuracy(double acc) { best_accuracy = acc; }
        void set_patience(size_t p) { patience = p; }
        void set_factor(double f) { factor = f; }
        void set_min_lr(double mlr) { min_lr = mlr; }
        void set_min_delta(double md) { min_delta = md; }
};
