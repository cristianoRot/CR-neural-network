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
    
    network.train(dataset, 200);
    network.save("checkpoints/model.crnn");
    
    return 0;
}

