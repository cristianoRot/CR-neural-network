// layer.cpp

#include "Functions.hpp"
#include "Layer.hpp"
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

    dA(),
    db(output_size, 1),
    dW(output_size, input_size),
    dZ(),

    vW(output_size, input_size),
    vb(output_size, 1),
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

void Layer::step(double lr, double beta)
{
    vW = (vW * beta) + (dW * (1 - beta));
    vb = (vb * beta) + (db * (1 - beta));

    W -= vW * lr;
    b -= vb * lr;
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
    A = activation(Z);
}

void Layer::backprop()
{
    switch (activation_type)
    {
        case Activation::RELU:
            backprop_relu();
            break;
        case Activation::SOFTMAX:
            backprop_softmax();
            break;
        case Activation::LINEAR:
            backprop_linear();
            break;
        case Activation::SIGMOID:
            // TODO
            break;
    }
}

void Layer::backprop_relu()
{
    dZ = dA.hadamard(Z.drelu());
    dW = dZ * prev_A->transpose();
    db = dZ.sum_columns();

    if (prev_dA != nullptr)
    {
        Matrix temp = W.transpose() * dZ;
        *prev_dA = temp;
    }
}

void Layer::backprop_softmax()
{
    dW = dZ * prev_A->transpose();
    db = dZ.sum_columns();

    if (prev_dA != nullptr)
    {
        Matrix temp = W.transpose() * dZ;
        *prev_dA = temp;
    }
}

void Layer::backprop_linear()
{
    dZ = dA;
    dW = dZ * prev_A->transpose();
    db = dZ.sum_columns();
    
    if (prev_dA != nullptr)
    {
        Matrix temp = W.transpose() * dZ;
        *prev_dA = temp;
    }
}

Matrix Layer::activation(const Matrix& Z)
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