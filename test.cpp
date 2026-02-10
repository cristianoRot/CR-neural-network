#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include <iostream>
#include <vector>

int main() 
{
    Dataset dataset = Dataset::from_csv("data/mnist_test.csv", {"ALL"}, "label");
    dataset.scale(1.0 / 255.0);

    Network network("models/mnist_model.crnn");

    Metrics metrics = network.eval(dataset);
    double accuracy = metrics.accuracy;
    std::cout << "\033[34mAccuracy: " << accuracy * 100 << "%\033[0m" << std::endl;

    std::cout << "\nConfusion Matrix:\n";
    metrics.confusion_matrix.print();
    
    return 0;
}

