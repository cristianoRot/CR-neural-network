#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include "include/Matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    Dataset dataset = Dataset::from_csv("data/btc_data.csv", {"ALL"}, "label");

    Network network(
        {
            Layer(24, 12, Activation::RELU),
            Layer(12, 6, Activation::RELU),
            Layer(6, 2, Activation::SOFTMAX)
        },
        InitType::He,
        0.01
    );
    
    network.train(dataset, 50);
    
    return 0;
}
