#include "Optimizer.hpp"
#include "Network.hpp"
#include "ModelIO.hpp"

Optimizer::Optimizer(double initial_lr, double min_lr, double factor, double min_delta, size_t patience)
    : learning_rate(initial_lr), 
        min_lr(min_lr), 
        factor(factor), 
        min_delta(min_delta), 
        patience(patience)
{ }

void Optimizer::step(std::vector<Layer>& layers)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].step(learning_rate, momentum);
    }
}

void Optimizer::lr_reduce_on_plateau(double current_accuracy, Network& network)
{
    if (current_accuracy > best_accuracy + min_delta)
    {
        best_accuracy = current_accuracy;
        patience_counter = 0;
        
        network.save("checkpoints/model.crnn");
        
        return;
    }

    patience_counter++;
    
    if (patience_counter >= patience)
    {
        double new_lr = learning_rate * factor;
        
        if (new_lr >= min_lr)
        {
            learning_rate = new_lr;            
            best_accuracy = current_accuracy;
        }
        
        patience_counter = 0;
    }
}
