#include "ModelIO.hpp"
#include "Network.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <cstring>

static bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// ---------------------------------------------------------
// Matrix I/O
// ---------------------------------------------------------

void ModelIO::write_matrix(std::ofstream& file, const Matrix& matrix)
{
    size_t rows = matrix.rows();
    size_t cols = matrix.cols();
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    const std::vector<double>& data = matrix.get_data();
    
    if (data.size() != rows * cols) throw std::runtime_error("Matrix data size mismatch");
    
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}

Matrix ModelIO::read_matrix(std::ifstream& file)
{
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    
    if (file.fail() || rows == 0 || cols == 0 || rows > 100000 || cols > 100000) {
        throw std::runtime_error("Invalid matrix dimensions");
    }
    
    std::vector<double> data(rows * cols);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(double));
    
    if (file.fail()) throw std::runtime_error("Failed to read matrix data.");
    
    return Matrix(rows, cols, data);
}

// ---------------------------------------------------------
// Layer I/O
// ---------------------------------------------------------

void ModelIO::write_layer(std::ofstream& file, const Layer& layer)
{
    size_t in = layer.get_input_size();
    size_t out = layer.get_output_size();
    int act = static_cast<int>(layer.get_activation());
    bool bn = layer.is_batch_norm();
    
    file.write(reinterpret_cast<const char*>(&in), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&out), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&act), sizeof(int));
    file.write(reinterpret_cast<const char*>(&bn), sizeof(bool));
    
    double dropout = layer.get_dropout_rate();
    file.write(reinterpret_cast<const char*>(&dropout), sizeof(double));

    write_matrix(file, layer.getW());
    write_matrix(file, layer.getb());
    write_matrix(file, layer.getvW());
    write_matrix(file, layer.getvb());
    
    if (bn) {
        write_matrix(file, layer.get_gamma());
        write_matrix(file, layer.get_running_mean());
        write_matrix(file, layer.get_running_var());
        write_matrix(file, layer.get_vGamma());
    }
}

Layer ModelIO::read_layer(std::ifstream& file)
{
    size_t in, out;
    int act_int;
    bool bn;
    
    file.read(reinterpret_cast<char*>(&in), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&out), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&act_int), sizeof(int));
    file.read(reinterpret_cast<char*>(&bn), sizeof(bool));
    
    if (file.fail()) throw std::runtime_error("Failed to read layer metadata.");

    Activation act = static_cast<Activation>(act_int);
    
    double dropout = 0.0;
    file.read(reinterpret_cast<char*>(&dropout), sizeof(double));
    
    Layer layer(in, out, act, dropout, bn);
    
    Matrix W = read_matrix(file);
    Matrix b = read_matrix(file);
    Matrix vW = read_matrix(file);
    Matrix vb = read_matrix(file);
    
    layer.setW(W);
    layer.setb(b);
    layer.setvW(vW);
    layer.setvb(vb);
    
    if (bn) {
        Matrix gamma = read_matrix(file);
        Matrix rm = read_matrix(file);
        Matrix rv = read_matrix(file);
        Matrix vGamma = read_matrix(file);
        
        layer.set_gamma(gamma);
        layer.set_running_mean(rm);
        layer.set_running_var(rv);
        layer.set_vGamma(vGamma);
    }
    
    return layer;
}

// ---------------------------------------------------------
// Network I/O
// ---------------------------------------------------------

void ModelIO::save_model(const Network& network, const std::string& filepath)
{
    // Create directory if needed (simple check)
    size_t last_slash = filepath.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        std::string dir = filepath.substr(0, last_slash);
        struct stat info;
        if (stat(dir.c_str(), &info) != 0) {
            std::string cmd = "mkdir -p " + dir;
            system(cmd.c_str());
        }
    }

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    file.write(MAGIC_HEADER, strlen(MAGIC_HEADER));
    int version = MODEL_VERSION;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));

    int loss = static_cast<int>(network.get_loss_type());
    size_t num_layers = network.get_layers().size();
    
    file.write(reinterpret_cast<const char*>(&loss), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    
    double lr = network.get_learning_rate();
    double best_acc = network.get_best_accuracy();
    size_t patience = network.get_patience();
    double factor = network.get_factor();
    double min_lr = network.get_min_lr();
    double min_delta = network.get_min_delta();

    file.write(reinterpret_cast<const char*>(&lr), sizeof(double));
    file.write(reinterpret_cast<const char*>(&best_acc), sizeof(double));
    file.write(reinterpret_cast<const char*>(&patience), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&factor), sizeof(double));
    file.write(reinterpret_cast<const char*>(&min_lr), sizeof(double));
    file.write(reinterpret_cast<const char*>(&min_delta), sizeof(double));

    // V3: Additional Optimizer State + current_epoch
    double momentum = network.get_momentum();
    size_t patience_counter = network.get_patience_counter();
    double clip = network.get_clipping_threshold();
    size_t epoch = network.get_current_epoch();

    file.write(reinterpret_cast<const char*>(&momentum), sizeof(double));
    file.write(reinterpret_cast<const char*>(&patience_counter), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&clip), sizeof(double));
    file.write(reinterpret_cast<const char*>(&epoch), sizeof(size_t));

    for (const auto& layer : network.get_layers()) {
        write_layer(file, layer);
    }

    file.close();
}

void ModelIO::load_model(Network& network, const std::string& filepath)
{
    if (!file_exists(filepath)) throw std::runtime_error("File not found: " + filepath);

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file for reading: " + filepath);

    char magic[16];
    memset(magic, 0, 16);
    file.read(magic, strlen(MAGIC_HEADER));
    if (strcmp(magic, MAGIC_HEADER) != 0) throw std::runtime_error("Invalid file format");
    
    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != MODEL_VERSION) throw std::runtime_error("Unsupported version: " + std::to_string(version) + ". Expected: " + std::to_string(MODEL_VERSION));

    int loss_int;
    size_t num_layers;
    
    file.read(reinterpret_cast<char*>(&loss_int), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    double lr, best_acc, factor, min_lr, min_delta;
    size_t patience;

    file.read(reinterpret_cast<char*>(&lr), sizeof(double));
    file.read(reinterpret_cast<char*>(&best_acc), sizeof(double));
    file.read(reinterpret_cast<char*>(&patience), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&factor), sizeof(double));
    file.read(reinterpret_cast<char*>(&min_lr), sizeof(double));
    file.read(reinterpret_cast<char*>(&min_delta), sizeof(double));
    
    Loss loss = static_cast<Loss>(loss_int);
    
    bool is_empty = network.get_layers().empty();
    if (is_empty) {
        network.set_loss_type(loss);
    } else if (network.get_loss_type() != loss) {
        throw std::runtime_error("Loss type mismatch in loaded model.");
    }
    
    network.set_learning_rate(lr);
    network.set_best_accuracy(best_acc);
    network.set_patience(patience);
    network.set_factor(factor);
    network.set_min_lr(min_lr);
    // Default V1/V2 values not supported anymore since we enforce strict version matching for struct consistency
    // But we still read the file linearly.

    double momentum, clip;
    size_t pc, epoch;

    file.read(reinterpret_cast<char*>(&momentum), sizeof(double));
    file.read(reinterpret_cast<char*>(&pc), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&clip), sizeof(double));
    file.read(reinterpret_cast<char*>(&epoch), sizeof(size_t));

    network.set_momentum(momentum);
    network.set_patience_counter(pc);
    network.set_clipping_threshold(clip);
    network.set_current_epoch(epoch);

    if (is_empty) {
        for (size_t i = 0; i < num_layers; ++i) {
            network.get_layers().push_back(read_layer(file));
        }
        for (size_t i = 1; i < network.get_layers().size(); ++i) {
            network.get_layers()[i].connect_prev(network.get_layers()[i-1]);
        }
    } else {
        if (network.get_layers().size() != num_layers) throw std::runtime_error("Layer count mismatch");
        
        for (size_t i = 0; i < num_layers; ++i) {
            Layer loaded = read_layer(file);
            Layer& current = network.get_layers()[i];
            
            if (current.get_input_size() != loaded.get_input_size() ||
                current.get_output_size() != loaded.get_output_size() ||
                current.get_activation() != loaded.get_activation() ||
                current.is_batch_norm() != loaded.is_batch_norm()) 
            {
                throw std::runtime_error("Layer " + std::to_string(i) + " architecture mismatch.");
            }
            
            current.setW(loaded.getW());
            current.setb(loaded.getb());
            current.setvW(loaded.getvW());
            current.setvb(loaded.getvb());
            
            if (current.is_batch_norm()) {
                current.set_gamma(loaded.get_gamma());
                current.set_running_mean(loaded.get_running_mean());
                current.set_running_var(loaded.get_running_var());
                current.set_vGamma(loaded.get_vGamma());
            }
        }
    }

    file.close();
}
