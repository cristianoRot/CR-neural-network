// layer.cpp

#include "Layer.hpp"
#include <random>
#include <cmath>

// Constructor

Layer::Layer(size_t input_size, size_t output_size, const Matrix* prev_A, Matrix* prev_dA) :
    input_size(input_size),
    output_size(output_size),

    A(output_size, 1),
    b(output_size, 1), 
    W(output_size, input_size),
    Z(output_size, 1),

    dA(output_size, 1),
    db(output_size, 1),
    dW(output_size, input_size),
    dZ(output_size, 1),

    vW(output_size, input_size),
    vb(output_size, 1),

    prev_A(prev_A),
    prev_dA(prev_dA)
{ }

HiddenLayer::HiddenLayer(size_t input_size, size_t output_size, const Matrix* prev_A, Matrix* prev_dA) :
    Layer(input_size, output_size, prev_A, prev_dA) { }

OutputLayer::OutputLayer(size_t input_size, size_t output_size, const Matrix* prev_A, Matrix* prev_dA) :
    Layer(input_size, output_size, prev_A, prev_dA) { }

// Getters and Setters

const Matrix& Layer::getA() const { return A; }
Matrix& Layer::getA() { return A; }
const Matrix& Layer::get_dA() const { return dA; }
Matrix& Layer::get_dA() { return dA; }

void Layer::setA(const Matrix& g) { A = g; }
void Layer::set_dA(const Matrix& g) { dA = g; }

const Matrix& Layer::get_dZ() const { return dZ; }
void Layer::set_dZ(const Matrix& g) { dZ = g; }

void Layer::step(double lr, double beta)
{
    vW = (vW * beta) + (dW * (1 - beta));
    vb = (vb * beta) + (db * (1 - beta));

    W -= vW * lr;
    b -= vb * lr;
}

// Hidden Layer

void Layer::init_weights(InitType init_type)
{
    b.fill(0.0);
    vW.fill(0.0);
    vb.fill(0.0);

    static thread_local std::mt19937 gen{ std::random_device{}() };

    const size_t fan_in  = input_size;
    const size_t fan_out = output_size;

    switch (init_type)
    {
        case InitType::Zero:
            W.fill(0.0);
            break;

        case InitType::Rand:
        {
            std::uniform_real_distribution<double> dist(-0.01, 0.01);
            for (std::size_t r = 0; r < output_size; ++r)
                for (std::size_t c = 0; c < input_size; ++c)
                    W.set(r, c, dist(gen));
            break;
        }

        case InitType::He:
        {
            const double stddev = std::sqrt(2.0 / static_cast<double>(fan_in));
            std::normal_distribution<double> dist(0.0, stddev);
            for (std::size_t r = 0; r < output_size; ++r)
                for (std::size_t c = 0; c < input_size; ++c)
                    W.set(r, c, dist(gen));
            break;
        }
    }
}

void HiddenLayer::forward()
{
    Z = (W * (*prev_A)) + b;
    A = Z.relu();
}

void HiddenLayer::backprop()
{
    dZ = dA.hadamard(Z.drelu());
    dW = dZ * prev_A->transpose();
    db = dZ;

    if (prev_dA != nullptr)
    {
        Matrix temp = W.transpose() * dZ;
        *prev_dA = temp;
    }
}

// Output Layer

void OutputLayer::forward()
{
    Z = (W * (*prev_A)) + b;
    A = Z.softmax();
}

void OutputLayer::backprop()
{
    dW = dZ * prev_A->transpose();
    db = dZ;

    if (prev_dA != nullptr)
    {
        Matrix temp = W.transpose() * dZ;
        *prev_dA = temp;
    }
}