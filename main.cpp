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
            Layer(4, 1, Activation::SIGMOID)
        },
        0.1,
        InitType::He,
        Loss::MSE
    );
    
    network.train(dataset, 10, 1);
    network.save("models/xor_model.crnn");
    
    return 0;
}