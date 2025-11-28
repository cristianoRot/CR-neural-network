// layer.hpp

#pragma once
#include "Functions.hpp"
#include "Matrix.hpp"

class Layer
{
    protected:
        size_t input_size;
        size_t output_size;

        Matrix A;
        Matrix b;
        Matrix W;
        Matrix Z;

        Matrix dA;
        Matrix db;
        Matrix dW;
        Matrix dZ;

        Matrix vb;
        Matrix vW;

        const Matrix* prev_A;
        Matrix* prev_dA;
        
    public:
        Layer(size_t input_size, size_t output_size);
        ~Layer() = default;

        void init_weights(InitType init_type);
        void connect_prev(const Layer& prev);

        // Getters
        const Matrix& getA() const;
        const Matrix& get_dA() const;
        Matrix& getA();
        Matrix& get_dA();
        const Matrix& get_dZ() const;

        // Setters
        void setA(const Matrix& g);
        void set_dA(const Matrix& g);
        void set_dZ(const Matrix& g);

        virtual void forward() = 0;
        virtual void backprop() = 0;

        void step(double lr, double beta);
};

class InputLayer final : public Layer
{
    public:
        InputLayer(size_t input_size, size_t output_size);

        void forward() override;
        void backprop() override;
};

class HiddenLayer final : public Layer
{
    public:
        HiddenLayer(size_t input_size, size_t output_size);

        void forward() override;
        void backprop() override;
};

class OutputLayer : public Layer
{
    public:
        OutputLayer(size_t input_size, size_t output_size);

        void forward() override;
        void backprop() override;
};
