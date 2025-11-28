#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include "include/Matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    Dataset dataset = Dataset::from_csv("data/btc_data.csv", {"ALL"}, "label");

    Network network(
        {
            Layer(24, 64, Activation::RELU),
            Layer(64, 32, Activation::RELU),
            Layer(32, 2, Activation::SOFTMAX)
        },
        0.001,
        InitType::He,
        Loss::CROSS_ENTROPY
    );
    
    network.load("checkpoints/best_btc.crnn");
    
    size_t correct = 0;

    for (size_t i = 0; i < dataset.size(); i++) 
    {
        network.forward(dataset.get_input(i));
        const Matrix& output = network.get_output();
        size_t predicted = network.argmax(output);
        size_t actual = dataset.get_output(i);
        
        if (predicted == actual) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / dataset.size();
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    return 0;
}
