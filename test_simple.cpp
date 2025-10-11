#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include "include/Matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<Matrix> inputs;
    std::vector<size_t> outputs;
    
    inputs.push_back(Matrix(2, 1, {0.0, 0.0}));
    outputs.push_back(0);
    inputs.push_back(Matrix(2, 1, {0.0, 1.0}));
    outputs.push_back(1);
    inputs.push_back(Matrix(2, 1, {1.0, 0.0}));
    outputs.push_back(1);
    inputs.push_back(Matrix(2, 1, {1.0, 1.0}));
    outputs.push_back(0);
    
    Dataset dataset(inputs, outputs);
    Network network({2, 2, 2}, InitType::He, 0.01);
    
    std::cout << "Starting training..." << std::endl;
    network.train(dataset, 100);
    
    std::cout << "Training completed!" << std::endl;
    return 0;
}
