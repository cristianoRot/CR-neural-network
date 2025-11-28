#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include "include/Matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    Dataset dataset = Dataset::from_csv("data/iris.csv", {"sepal_length","sepal_width","petal_length","petal_width"}, "species");
    Network network(
        {
            Layer(4, 4, Activation::RELU),
            Layer(4, 3, Activation::SOFTMAX)
        },
        InitType::He,
        0.01
    );
    
    std::cout << "Starting training..." << std::endl;
    network.train(dataset, 1000);
    
    std::cout << "\nTraining completed!" << std::endl;
    
    return 0;
}
