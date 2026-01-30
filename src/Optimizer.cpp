#include "Optimizer.hpp"
#include "Network.hpp"
#include "ModelIO.hpp"
#include <iostream>

Optimizer::Optimizer(double initial_lr, double min_lr, double factor, double min_delta, size_t patience)
    : learning_rate(initial_lr), 
        min_lr(min_lr), 
        factor(factor), 
        min_delta(min_delta), 
        patience(patience)
{ }

void Optimizer::step(std::vector<Layer>& layers)
{
    // Gradient Clipping
    apply_clipping(layers);

    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].step(learning_rate, momentum);
    }
}

void Optimizer::apply_clipping(std::vector<Layer>& layers) {
    double sum_sq = 0.0;
    
    for (const auto& layer : layers) 
    {
        sum_sq += layer.get_sq_grad_sum();
    }
    
    double norm = std::sqrt(sum_sq);

    if (norm > clipping_threshold) 
    {
        double scale = clipping_threshold / norm;
        for (auto& layer : layers) {
            layer.scale_gradients(scale);
        }
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
            std::cout << " [Plateau] Reducing LR from " << learning_rate << " to " << new_lr << std::endl;
            learning_rate = new_lr;            
        }
        
        patience_counter = 0;
    }
}