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
    Network network({2, 4, 2}, InitType::He, 0.1);
    
    std::cout << "Starting training..." << std::endl;
    network.train(dataset, 100);
    
    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "\nTesting final predictions:" << std::endl;
    
    for (size_t i = 0; i < dataset.size(); i++) 
    {
        network.forward(dataset.get_input(i));
        const Matrix& output = network.get_output();
        size_t predicted = network.argmax(output);
        size_t expected = dataset.get_output(i);
        
        std::cout << "Input: [" << dataset.get_input(i).get(0, 0) 
                  << ", " << dataset.get_input(i).get(1, 0) << "] -> "
                  << "Predicted: " << predicted << " (Expected: " << expected << ") "
                  << (predicted == expected ? "✓" : "✗") << std::endl;
    }
    
    return 0;
}
