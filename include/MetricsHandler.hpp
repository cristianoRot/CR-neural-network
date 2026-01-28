#pragma once
#include "Matrix.hpp"
#include "Functions.hpp"
#include <vector>
#include <cstddef>

class MetricsHandler
{
public:
    static Matrix compute_loss_gradient(const Matrix& prediction, const std::vector<size_t>& labels, Loss loss_type, const std::vector<double>& class_weights);
    static double accumulate_loss(const Matrix& prediction, const std::vector<size_t>& labels, double current_accumulated_loss, Loss loss_type, const std::vector<double>& class_weights);
    static size_t count_correct_predictions(const Matrix& prediction, const std::vector<size_t>& labels);
};
