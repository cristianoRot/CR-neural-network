#include "MetricsHandler.hpp"
#include "Layer.hpp"
#include <cmath>
#include <stdexcept>

// compute dA (and dZ in CE + softmax) for the last layer
void MetricsHandler::compute_loss_gradient(Layer& last_layer, const std::vector<size_t>& labels, Loss loss_type, const std::vector<double>& class_weights)
{
    const Matrix& prediction = last_layer.getA();
    const size_t batch_size = prediction.cols();
    const size_t num_classes = prediction.rows();

    if (labels.size() != batch_size)
    {
        throw std::invalid_argument("Error: Labels must have the same size as the batch size");
    }

    switch (loss_type)
    {
        case Loss::CROSS_ENTROPY:
        {
            if (last_layer.get_activation() == Activation::SOFTMAX) 
                 compute_ce_softmax_gradient(last_layer, labels, class_weights);
            else compute_ce_general_gradient(last_layer, labels, class_weights);
            break;
        }
        case Loss::MSE:
        {
            Matrix dLoss_dA(num_classes, batch_size);

            // MSE derivative: 2 * (A - Y)
            for (size_t i = 0; i < batch_size; ++i)
            {
                size_t label = labels[i];
                double weight = class_weights[label];
                
                for (size_t cls = 0; cls < num_classes; ++cls)
                {
                    double target = (cls == label) ? 1.0 : 0.0;
                    double pred   = prediction.get(cls, i);
                    double diff   = pred - target;
                    dLoss_dA.set(cls, i, 2.0 * diff * weight);
                }
            }

            dLoss_dA /= batch_size;
            last_layer.set_dA(dLoss_dA);
            break;
        }
    }
}

double MetricsHandler::accumulate_loss(const Matrix& prediction, const std::vector<size_t>& labels, double current_accumulated_loss, Loss loss_type, const std::vector<double>& class_weights)
{
    const size_t B = prediction.cols();
    double batch_loss = 0.0;

    switch (loss_type)
    {
        case Loss::CROSS_ENTROPY:
        {
            for (size_t i = 0; i < B; i++)
            {
                size_t label = labels[i];
                double weight = class_weights[label];

                double pred_prob = prediction.get(label, i);
                if (pred_prob < 1e-10) pred_prob = 1e-10; // Avoid log(0)
                batch_loss -= weight * std::log(pred_prob);
            }
            break;
        }
        case Loss::MSE:
        {
            const size_t C = prediction.rows();

            for (size_t i = 0; i < B; ++i)
            {
                size_t label = labels[i];
                double weight = class_weights[label];

                double mse_sample = 0.0;

                for (size_t cls = 0; cls < C; ++cls)
                {
                    double target = (cls == label) ? 1.0 : 0.0;
                    double pred   = prediction.get(cls, i);
                    double diff   = pred - target;
                    mse_sample   += weight * diff * diff;
                }

                batch_loss += mse_sample;
            }
            break;
        }
    }

    return current_accumulated_loss + batch_loss;
}

size_t MetricsHandler::count_correct_predictions(const Matrix& prediction, const std::vector<size_t>& labels)
{
    size_t correct = 0;
    const size_t B = prediction.cols();

    for (size_t i = 0; i < B; i++)
    {
        size_t label = labels[i];
        size_t argmax = prediction.argmax_col(i);

        if (argmax == label) correct++;
    }

    return correct;
}

void MetricsHandler::compute_ce_softmax_gradient(Layer& last_layer, const std::vector<size_t>& labels, const std::vector<double>& class_weights)
{
    Matrix dZ = last_layer.getA();
    const size_t batch_size = dZ.cols();

    for (size_t i = 0; i < batch_size; i++)
    {
        size_t label = labels[i];
        double weight = class_weights[label];

        double v = dZ.get(label, i);
        dZ.set(label, i, v - 1.0);
        dZ.multiply_col(i, weight);
    }

    dZ /= batch_size;
    last_layer.set_dZ(dZ);
}

void MetricsHandler::compute_ce_general_gradient(Layer& last_layer, const std::vector<size_t>& labels, const std::vector<double>& class_weights)
{
    const Matrix& prediction = last_layer.getA();
    const size_t batch_size = prediction.cols();
    const size_t num_classes = prediction.rows();

    Matrix dA(num_classes, batch_size);

    for (size_t i = 0; i < batch_size; i++)
    {
        size_t label = labels[i];
        double weight = class_weights[label];
        
        double pred = prediction.get(label, i);
        if (pred < 1e-10) pred = 1e-10;

        dA.set(label, i, -weight / pred);
    }

    dA /= batch_size;
    last_layer.set_dA(dA);
}
