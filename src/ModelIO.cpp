#include "ModelIO.hpp"
#include "Network.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>

void ModelIO::write_matrix(std::ofstream& file, const Matrix& matrix)
{
    size_t rows = matrix.rows();
    size_t cols = matrix.cols();
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    const std::vector<double>& data = matrix.get_data();
    file.write(reinterpret_cast<const char*>(data.data()), rows * cols * sizeof(double));
}

Matrix ModelIO::read_matrix(std::ifstream& file)
{
    size_t rows, cols;
    
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    
    std::vector<double> data(rows * cols);
    file.read(reinterpret_cast<char*>(data.data()), rows * cols * sizeof(double));
    
    return Matrix(rows, cols, data);
}

void ModelIO::write_layer(std::ofstream& file, const Layer& layer)
{
    size_t input_size = layer.get_input_size();
    size_t output_size = layer.get_output_size();
    int activation = static_cast<int>(layer.get_activation());
    
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&output_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&activation), sizeof(int));
    
    write_matrix(file, layer.getW());
    write_matrix(file, layer.getb());
    write_matrix(file, layer.getvW());
    write_matrix(file, layer.getvb());
}

Layer ModelIO::read_layer(std::ifstream& file)
{
    size_t input_size, output_size;
    int activation_int;
    
    file.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&output_size), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&activation_int), sizeof(int));
    
    Activation activation = static_cast<Activation>(activation_int);
    
    Layer layer(input_size, output_size, activation);
    
    Matrix W = read_matrix(file);
    Matrix b = read_matrix(file);
    Matrix vW = read_matrix(file);
    Matrix vb = read_matrix(file);
    
    layer.setW(W);
    layer.setb(b);
    layer.setvW(vW);
    layer.setvb(vb);
    
    return layer;
}

void ModelIO::save_model(const Network& network, const std::string& filepath)
{
    size_t last_slash = filepath.find_last_of("/\\");
    if (last_slash != std::string::npos)
    {
        std::string dir = filepath.substr(0, last_slash);
        struct stat info;
        if (stat(dir.c_str(), &info) != 0)
        {
            std::string mkdir_cmd = "mkdir -p " + dir;
            int result = system(mkdir_cmd.c_str());
            if (result != 0)
            {
                throw std::runtime_error("Error: Cannot create directory: " + dir);
            }
        }
    }
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: Cannot open file for writing: " + filepath);
    }
    
    if (!file.good())
    {
        file.close();
        throw std::runtime_error("Error: File stream is not in good state: " + filepath);
    }
    
    const std::vector<Layer>& layers = network.get_layers();
    size_t num_layers = layers.size();
    
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    
    for (const auto& layer : layers)
    {
        write_layer(file, layer);
    }
    
    int loss_type_int = static_cast<int>(network.get_loss_type());
    double learning_rate = network.get_learning_rate();
    double best_accuracy = network.get_best_accuracy();
    size_t patience = network.get_patience();
    double factor = network.get_factor();
    double min_lr = network.get_min_lr();
    double min_delta = network.get_min_delta();
    
    file.write(reinterpret_cast<const char*>(&loss_type_int), sizeof(int));
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(double));
    file.write(reinterpret_cast<const char*>(&best_accuracy), sizeof(double));
    file.write(reinterpret_cast<const char*>(&patience), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&factor), sizeof(double));
    file.write(reinterpret_cast<const char*>(&min_lr), sizeof(double));
    file.write(reinterpret_cast<const char*>(&min_delta), sizeof(double));
    
    if (!file.good())
    {
        file.close();
        throw std::runtime_error("Error: Failed to write data to file: " + filepath);
    }
    
    file.close();
    
    if (!file.good() && !file.eof())
    {
        throw std::runtime_error("Error: Failed to close file properly: " + filepath);
    }
    
    std::cout << "Model saved to: " << filepath << std::endl;
}

void ModelIO::load_model(Network& network, const std::string& filepath)
{
    struct stat info;
    if (stat(filepath.c_str(), &info) != 0)
    {
        throw std::runtime_error("Error: File does not exist: " + filepath);
    }
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: Cannot open file for reading: " + filepath);
    }
    
    if (!file.good())
    {
        file.close();
        throw std::runtime_error("Error: File stream is not in good state: " + filepath);
    }
    
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    if (num_layers != network.get_layers().size())
    {
        file.close();
        throw std::runtime_error("Error: Number of layers mismatch. Expected " + 
                                 std::to_string(network.get_layers().size()) + ", found " + 
                                 std::to_string(num_layers));
    }
    
    for (size_t i = 0; i < num_layers; i++)
    {
        size_t input_size, output_size;
        int activation_int;
        
        file.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&activation_int), sizeof(int));
        
        if (input_size != network.get_layers()[i].get_input_size() || 
            output_size != network.get_layers()[i].get_output_size() ||
            static_cast<Activation>(activation_int) != network.get_layers()[i].get_activation())
        {
            file.close();
            throw std::runtime_error("Error: Layer " + std::to_string(i) + " architecture mismatch");
        }
        
        Matrix W = read_matrix(file);
        Matrix b = read_matrix(file);
        Matrix vW = read_matrix(file);
        Matrix vb = read_matrix(file);
        
        network.get_layers()[i].setW(W);
        network.get_layers()[i].setb(b);
        network.get_layers()[i].setvW(vW);
        network.get_layers()[i].setvb(vb);
    }
    
    int loss_type_int;
    double loaded_learning_rate;
    double loaded_best_accuracy;
    size_t loaded_patience;
    double loaded_factor;
    double loaded_min_lr;
    double loaded_min_delta;
    
    file.read(reinterpret_cast<char*>(&loss_type_int), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_learning_rate), sizeof(double));
    file.read(reinterpret_cast<char*>(&loaded_best_accuracy), sizeof(double));
    file.read(reinterpret_cast<char*>(&loaded_patience), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&loaded_factor), sizeof(double));
    file.read(reinterpret_cast<char*>(&loaded_min_lr), sizeof(double));
    file.read(reinterpret_cast<char*>(&loaded_min_delta), sizeof(double));
    
    if (!file.good() && !file.eof())
    {
        file.close();
        throw std::runtime_error("Error: Failed to read data from file: " + filepath);
    }
    
    file.close();
    
    if (static_cast<Loss>(loss_type_int) != network.get_loss_type())
    {
        throw std::runtime_error("Error: Loss type mismatch");
    }
    
    network.set_learning_rate(loaded_learning_rate);
    network.set_best_accuracy(loaded_best_accuracy);
    network.set_patience(loaded_patience);
    network.set_factor(loaded_factor);
    network.set_min_lr(loaded_min_lr);
    network.set_min_delta(loaded_min_delta);
    
    std::cout << "Model loaded from: " << filepath << std::endl;
}
