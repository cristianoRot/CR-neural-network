#include "Network.hpp"
#include "Dataset.hpp"
#include "RNG.hpp"
#include <vector>

int main() 
{
    Dataset train_dataset = Dataset::from_csv("data/mnist_train.csv", {"ALL"}, "label");
    Dataset eval_dataset = Dataset::from_csv("data/mnist_eval.csv", {"ALL"}, "label");
    
    train_dataset.scale(1.0 / 255.0);
    eval_dataset.scale(1.0 / 255.0);

    RNG::set_seed(42);

    Network network(
        {
            Layer(784, 512, Activation::RELU, 0.2),
            Layer(512, 128, Activation::RELU, 0.2),
            Layer(128, 10,  Activation::SOFTMAX)
        },
        0.1,
        InitType::He,
        Loss::CROSS_ENTROPY
    );
    
    network.train(train_dataset, eval_dataset, 100, 64);
    network.save_best("models/mnist_model.crnn");
    
    return 0;
}