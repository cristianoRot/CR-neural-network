// layer.cpp

#include "Functions.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "RNG.hpp"
#include <cmath>

// Constructor

Layer::Layer(size_t input_size, size_t output_size, Activation activation_type) :
    input_size(input_size),
    output_size(output_size),
    activation_type(activation_type),

    A(),
    b(output_size, 1), 
    W(output_size, input_size),
    Z(),
    Z_norm(),

    dA(),
    db(output_size, 1),
    dW(output_size, input_size),
    dZ(),

    // Batch Norm parameters
    gamma(output_size, 1),
    beta(output_size, 1),
    dgamma(output_size, 1),
    dbeta(output_size, 1),

    running_mean(output_size, 1),
    running_var(output_size, 1),

    vW(output_size, input_size),
    vb(output_size, 1),
    vGamma(output_size, 1),
    vBeta(output_size, 1),

    prev_A(nullptr),
    prev_dA(nullptr)
{ }

// Getters and Setters

const Matrix& Layer::getA() const { return A; }
const Matrix& Layer::get_dA() const { return dA; }

Matrix& Layer::getA() { return A; }
Matrix& Layer::get_dA() { return dA; }

void Layer::setA(const Matrix& g) { A = g; }
void Layer::set_dA(const Matrix& g) { dA = g; }

const Matrix& Layer::get_dZ() const { return dZ; }
void Layer::set_dZ(const Matrix& g) { dZ = g; }

void Layer::set_prev_A(const Matrix* prev_A_ptr) { prev_A = prev_A_ptr; }

void Layer::step(double lr, double momentum)
{
    vW = (vW * momentum) + (dW);
    vb = (vb * momentum) + (db);

    W -= vW * lr;
    b -= vb * lr;

    vGamma = (vGamma * momentum) + dgamma;
    vBeta  = (vBeta * momentum)  + dbeta;

    gamma -= vGamma * lr;
    beta  -= vBeta * lr;
}


// Connectors

void Layer::connect_prev(const Layer& prev)
{
    if (prev.output_size != this->input_size)
    {
        throw std::invalid_argument(
            "Error: Dimension mismatch in layer connection. "
            "Previous layer output size (" + std::to_string(prev.output_size) + 
            ") does not match current layer input size (" + std::to_string(this->input_size) + ")"
        );
    }

    prev_A = &prev.getA();
    prev_dA = const_cast<Matrix*>(&prev.get_dA());
}

// Hidden Layer

void Layer::init_weights(InitType init_type)
{
    b.fill(0.0);
    vW.fill(0.0);
    vb.fill(0.0);
    vGamma.fill(0.0);
    vBeta.fill(0.0);
    
    // Batch Norm initialization
    gamma.fill(1.0);
    beta.fill(0.0);
    running_mean.fill(0.0);
    running_var.fill(1.0);

    const size_t fan_in  = input_size;
    const size_t fan_out = output_size;

    switch (init_type)
    {
        case InitType::Zero:
            W.fill(0.0);
            break;

        case InitType::Rand:
        {
            for (std::size_t r = 0; r < output_size; ++r)
                for (std::size_t c = 0; c < input_size; ++c)
                    W.set(r, c, RNG::get_random_range(-0.01, 0.01));
            break;
        }

        case InitType::He:
        {
            const double stddev = std::sqrt(2.0 / static_cast<double>(fan_in));
            for (std::size_t r = 0; r < output_size; ++r)
                for (std::size_t c = 0; c < input_size; ++c)
                    W.set(r, c, RNG::get_normal(0.0, stddev));
            break;
        }
    }
}

void Layer::forward()
{
    Z = W * (*prev_A);
    Z.add_col_vector(b);
    
    batch_norm_forward();

    A = activation();
}

void Layer::backprop()
{
    compute_dz();
    batch_norm_backprop();

    dW = dZ * prev_A->transpose();
    db = dZ.sum_columns();

    if (prev_dA == nullptr) return;
    
    Matrix temp = W.transpose() * dZ;
    *prev_dA = temp;
}

void Layer::batch_norm_forward()
{
    switch (mode)
    {
        case Mode::TRAIN:
        {
            batch_mean = Z.mean();
            batch_var = Z.variance(batch_mean);
            
            Z_norm = Z.normalize(batch_mean, batch_var);
            
            running_mean = running_mean * 0.9 + batch_mean * 0.1;
            running_var = running_var * 0.9 + batch_var * 0.1;
            break;
        }
        case Mode::EVAL:
            Z_norm = Z.normalize(running_mean, running_var);
            break;
    }

    Z = Z_norm; 
    Z.mul_col_vector(gamma);
    Z.add_col_vector(beta);
}

void Layer::batch_norm_backprop()
{
    if (mode != Mode::TRAIN) return;

    dgamma = dZ.hadamard(Z_norm).sum_columns();
    dbeta = dZ.sum_columns();
    
    dZ.mul_col_vector(gamma);
    dZ = dZ.normalize_derivative(batch_mean, batch_var, Z_norm);
}

Matrix Layer::activation()
{
    switch (activation_type)
    {
        case Activation::RELU:
            return Z.relu();
        case Activation::SOFTMAX:
            return Z.softmax();
        case Activation::LINEAR:
            return Z;
        case Activation::SIGMOID:
            // TODO: Implement sigmoid activation
            return Z;
    }
}

void Layer::compute_dz()
{
    switch (activation_type)
    {
        case Activation::RELU:
            dZ = dA.hadamard(Z.drelu());
            break;
        case Activation::LINEAR:
            dZ = dA;
            break;
        case Activation::SOFTMAX:
            // dZ already computed in MetricsHandler::compute_loss_gradient
            break;
        case Activation::SIGMOID:
            // TODO
            break;
    }
}

// Gradient clipping

double Layer::get_sq_grad_sum() const
{
    return dW.sum_of_squares() + db.sum_of_squares() + dgamma.sum_of_squares() + dbeta.sum_of_squares();
}

void Layer::scale_gradients(double scale)
{
    dW *= scale;
    db *= scale;
    dgamma *= scale;
    dbeta *= scale;
}

