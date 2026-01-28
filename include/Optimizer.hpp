#pragma once
#include <vector>
#include <cstddef>
#include "Layer.hpp"

class Network;

class Optimizer
{
    private:
        double learning_rate;
        double min_lr;
        double factor;
        double min_delta;
        size_t patience;
        
        size_t patience_counter = 0;
        double best_accuracy = 0.0;
        double momentum = 0.9;

    public:
        Optimizer(double initial_lr, double min_lr = 1e-6, double factor = 0.7, double min_delta = 0.001, size_t patience = 20);

        void step(std::vector<Layer>& layers);
        void lr_reduce_on_plateau(double current_accuracy, Network& network);

        double get_learning_rate() const { return learning_rate; }
        double get_best_accuracy() const { return best_accuracy; }
        size_t get_patience() const { return patience; }
        double get_factor() const { return factor; }
        double get_min_lr() const { return min_lr; }
        double get_min_delta() const { return min_delta; }
        
        void set_learning_rate(double lr) { learning_rate = lr; }
        void set_best_accuracy(double acc) { best_accuracy = acc; }
        void set_patience(size_t p) { patience = p; }
        void set_factor(double f) { factor = f; }
        void set_min_lr(double mlr) { min_lr = mlr; }
        void set_min_delta(double md) { min_delta = md; }
};
