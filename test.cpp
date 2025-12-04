#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include "include/Matrix.hpp"
#include <iostream>
#include <vector>

int main() 
{
    Dataset dataset = Dataset::from_csv("data/mnist_test.csv", {"ALL"}, "label");

    Network network(
        {
            Layer(784, 64, Activation::RELU),
            Layer(64, 32, Activation::RELU),
            Layer(32, 10,  Activation::SOFTMAX)
        },
        0.001,
        InitType::He,
        Loss::CROSS_ENTROPY
    );
    
    network.load("models/mnist_model.crnn");

    size_t correct = 0;

    for (size_t i = 0; i < dataset.size(); i++) 
    {
        Matrix input = dataset.get_input(i, 1);
        network.forward(input);
        const Matrix& output = network.get_output();
        size_t predicted = output.argmax_col(0);
        std::vector<size_t> labels = dataset.get_output(i, 1);
        size_t actual = labels[0];
        
        if (predicted == actual) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / dataset.size();
    std::cout << "\033[34mAccuracy: " << accuracy * 100 << "%\033[0m" << std::endl;
    
    return 0;
}

