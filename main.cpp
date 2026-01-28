#include "Network.hpp"
#include "Dataset.hpp"
#include "RNG.hpp"
#include <vector>

int main() 
{
    Dataset dataset = Dataset::from_csv("data/xor.csv", {"ALL"}, "label");

    RNG::set_seed(42);

    Network network(
        {
            Layer(2, 4, Activation::RELU),
            Layer(4, 2, Activation::SOFTMAX)
        },
        0.01,
        InitType::He,
        Loss::CROSS_ENTROPY
    );
    
    network.train(dataset, 100, 1);
    network.save("models/xor_model.crnn");
    
    return 0;
}