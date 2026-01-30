#include "include/Network.hpp"
#include "include/Dataset.hpp"
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

    Metrics metrics = network.eval(dataset);
    double accuracy = metrics.accuracy;
    std::cout << "\033[34mAccuracy: " << accuracy * 100 << "%\033[0m" << std::endl;
    
    return 0;
}

