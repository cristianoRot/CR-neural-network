#pragma once
#include "Network.hpp"
#include <string>
#include <fstream>
#include <vector>

class ModelIO {
public:
    static constexpr const char* MAGIC_HEADER = "CRNN_MODEL";
    static constexpr int MODEL_VERSION = 1;

    static void save_model(const Network& network, const std::string& filepath);
    static void load_model(Network& network, const std::string& filepath);
    
private:
    static void write_matrix(std::ofstream& file, const Matrix& matrix);
    static Matrix read_matrix(std::ifstream& file);
    
    static void write_layer(std::ofstream& file, const Layer& layer);
    static Layer read_layer(std::ifstream& file);
};

