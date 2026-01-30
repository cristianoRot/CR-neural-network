#include "Network.hpp"
#include "Dataset.hpp"
#include "RNG.hpp"
#include <vector>

int main() 
{
    Dataset dataset = Dataset::from_csv("data/mnist_train.csv", {"ALL"}, "label");
    dataset.scale(1.0 / 255.0);
    RNG::set_seed(42);

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
    
    network.train(dataset, 100, 32);
    network.save("models/mnist_model.crnn");
    
    return 0;
}

