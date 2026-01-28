#include "MetricsHandler.hpp"
#include <cmath>
#include <stdexcept>

Matrix MetricsHandler::compute_loss_gradient(const Matrix& prediction, const std::vector<size_t>& labels, Loss loss_type, const std::vector<double>& class_weights)
{
    const size_t batch_size = prediction.cols();
    const size_t num_classes = prediction.rows();

    if (labels.size() != batch_size)
    {
        throw std::invalid_argument("Error: Labels must have the same size as the batch size");
    }

    Matrix dZ;

    switch (loss_type)
    {
        case Loss::CROSS_ENTROPY:
        {
            dZ = prediction;

            for (size_t i = 0; i < batch_size; i++)
            {
                size_t label = labels[i];
                double weight = class_weights[label];

                double v = dZ.get(label, i);
                dZ.set(label, i, v - 1.0);
                dZ.multiply_col(i, weight);
            }

            dZ /= batch_size;
            break;
        }
        case Loss::MSE:
        {
            dZ = Matrix(num_classes, batch_size);

            for (size_t i = 0; i < batch_size; ++i)
            {
                size_t label = labels[i];
                double weight = class_weights[label];
                
                for (size_t cls = 0; cls < num_classes; ++cls)
                {
                    double target = (cls == label) ? 1.0 : 0.0;
                    double pred   = prediction.get(cls, i);
                    double diff   = pred - target;
                    dZ.set(cls, i, 2.0 * diff * weight);
                }
            }

            dZ /= batch_size;
            break;
        }
    }

    return dZ;
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
