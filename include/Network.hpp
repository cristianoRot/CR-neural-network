// network.hpp

#pragma once
#include "Functions.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Dataset.hpp"
#include "Optimizer.hpp"

struct Metrics {
    double accuracy;
    double loss;
};

class Network 
{
    private:
        Mode mode = Mode::TRAIN;
        
        std::vector<Layer> layers;
        std::vector<double> class_weights;

        Loss loss_type;
        Optimizer optimizer;

        size_t dataset_size = 0;

        Metrics do_pass(Dataset& dataset, size_t batch_size);

    public:
        Network(std::vector<Layer> layers, double learning_rate, InitType init_type, Loss loss_type = Loss::CROSS_ENTROPY);
        Network(const std::string& filepath);

        void init_weights(InitType init_type);
        void load(const std::string& filepath);
        void save(const std::string& filepath);
        void save_best(const std::string& filepath);
        const Matrix& get_output() const;

        Metrics eval(Dataset& dataset);
        void train(Dataset& train_dataset, Dataset& eval_dataset, size_t epochs, size_t batch_size);
        
        void forward(const Matrix& input);
        void backprop(const std::vector<size_t>& labels);
        
        // Model I/O getters
        std::vector<Layer>& get_layers() { return layers; }
        const std::vector<Layer>& get_layers() const { return layers; }
        Loss get_loss_type() const { return loss_type; }
        Mode get_mode() const { return mode; }
        
        // Proxy getters for Optimizer
        double get_learning_rate() const { return optimizer.get_learning_rate(); }
        double get_best_accuracy() const { return optimizer.get_best_accuracy(); }
        size_t get_patience() const { return optimizer.get_patience(); }
        double get_factor() const { return optimizer.get_factor(); }
        double get_min_lr() const { return optimizer.get_min_lr(); }
        double get_min_delta() const { return optimizer.get_min_delta(); }
        
        // Model I/O setters
        void set_mode(Mode m) { mode = m; }
        void set_loss_type(Loss l) { loss_type = l; }
        
        // Proxy setters for Optimizer
        void set_learning_rate(double lr) { optimizer.set_learning_rate(lr); }
        void set_best_accuracy(double acc) { optimizer.set_best_accuracy(acc); }
        void set_patience(size_t p) { optimizer.set_patience(p); }
        void set_factor(double f) { optimizer.set_factor(f); }
        void set_min_lr(double mlr) { optimizer.set_min_lr(mlr); }
        void set_min_delta(double md) { optimizer.set_min_delta(md); }
};
