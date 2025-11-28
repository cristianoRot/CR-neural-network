// layer.hpp

#pragma once
#include "Functions.hpp"
#include "Matrix.hpp"

class Layer
{
    protected:
        size_t input_size;
        size_t output_size;

        Activation activation;

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
        Layer(size_t input_size, size_t output_size, Activation activation);
        ~Layer() = default;

        void init_weights(InitType init_type);
        void connect_prev(const Layer& prev);

        // Getters
        const Matrix& getA() const;
        const Matrix& get_dA() const;
        const Matrix& get_dZ() const;

        Matrix& getA();
        Matrix& get_dA();

        // Setters
        void setA(const Matrix& g);
        void set_dA(const Matrix& g);
        void set_dZ(const Matrix& g);
        void set_prev_A(const Matrix* prev_A_ptr);
        
        // Model I/O getters
        size_t get_input_size() const { return input_size; }
        size_t get_output_size() const { return output_size; }
        Activation get_activation() const { return activation; }
        const Matrix& getW() const { return W; }
        const Matrix& getb() const { return b; }
        const Matrix& getvW() const { return vW; }
        const Matrix& getvb() const { return vb; }
        
        // Model I/O setters
        void setW(const Matrix& w) { W = w; }
        void setb(const Matrix& bias) { b = bias; }
        void setvW(const Matrix& vw) { vW = vw; }
        void setvb(const Matrix& vbias) { vb = vbias; }

        void forward();
        void backprop();

        void step(double lr, double beta);

    private:
        void backprop_relu();
        void backprop_softmax();
};
