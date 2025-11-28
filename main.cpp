#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include "include/Matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    Dataset dataset = Dataset::from_csv("data/mnist_train.csv", {"ALL"}, "label");

    Network network(
        {
            Layer(784, 128, Activation::RELU),
            Layer(128, 32, Activation::RELU),
            Layer(32, 10, Activation::SOFTMAX)
        },
        InitType::He,
        0.01
    );
    
    network.train(dataset, 10);
    
    return 0;
}
