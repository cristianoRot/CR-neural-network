#pragma once
#include "Matrix.hpp"
#include "Functions.hpp"
#include "Layer.hpp"
#include <vector>
#include <cstddef>

class MetricsHandler
{
public:
    static void compute_loss_gradient(Layer& last_layer, const std::vector<size_t>& labels, Loss loss_type, const std::vector<double>& class_weights);
    static double accumulate_loss(const Matrix& prediction, const std::vector<size_t>& labels, double current_accumulated_loss, Loss loss_type, const std::vector<double>& class_weights);
    static size_t count_correct_predictions(const Matrix& prediction, const std::vector<size_t>& labels);
    static void update_confusion_matrix(Matrix& confusion_matrix, const Matrix& prediction, const std::vector<size_t>& labels);

private:
    static void compute_ce_softmax_gradient(Layer& last_layer, const std::vector<size_t>& labels, const std::vector<double>& class_weights);
    static void compute_ce_general_gradient(Layer& last_layer, const std::vector<size_t>& labels, const std::vector<double>& class_weights);
};
