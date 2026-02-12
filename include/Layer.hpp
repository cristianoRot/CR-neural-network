// layer.hpp

#pragma once
#include "Functions.hpp"
#include "Matrix.hpp"

class Layer
{
    private:
        Mode mode = Mode::TRAIN;

        size_t input_size;
        size_t output_size;

        Activation activation_type;

        Matrix A;
        Matrix b;
        Matrix W;
        Matrix Z;
        Matrix Z_norm;

        Matrix dA;
        Matrix db;
        Matrix dW;
        Matrix dZ;

        Matrix vb;
        Matrix vW;
        Matrix vGamma;

        // Batch Norm parameters
        Matrix gamma;
        Matrix dgamma;

        Matrix running_mean;
        Matrix running_var;

        Matrix batch_mean;
        Matrix batch_var;

        // Dropout
        Matrix dropout_mask;
        double dropout_rate;

        const Matrix* prev_A;
        Matrix* prev_dA;
        
        bool use_batch_norm;

    public:
        Layer(size_t input_size, 
            size_t output_size, 
            Activation activation_type,
            double dropout_rate = 0.0, 
            bool use_batch_norm = true);

        ~Layer() = default;

        void init_weights(InitType init_type);
        void connect_prev(const Layer& prev);
        void set_mode(Mode m) { mode = m; }

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
        Activation get_activation() const { return activation_type; }
        double get_dropout_rate() const { return dropout_rate; }
        const Matrix& getW() const { return W; }
        const Matrix& getb() const { return b; }
        const Matrix& getvW() const { return vW; }
        const Matrix& getvb() const { return vb; }
        
        // Model I/O setters
        void setW(const Matrix& w) { W = w; }
        void setb(const Matrix& bias) { b = bias; }
        void setvW(const Matrix& vw) { vW = vw; }
        void setvb(const Matrix& vbias) { vb = vbias; }
        
        // Batch Norm & Model I/O helpers
        bool is_batch_norm() const { return use_batch_norm; }
        
        const Matrix& get_gamma() const { return gamma; }
        const Matrix& get_running_mean() const { return running_mean; }
        const Matrix& get_running_var() const { return running_var; }
        const Matrix& get_vGamma() const { return vGamma; }

        void set_gamma(const Matrix& m) { gamma = m; }
        void set_running_mean(const Matrix& m) { running_mean = m; }
        void set_running_var(const Matrix& m) { running_var = m; }
        void set_vGamma(const Matrix& m) { vGamma = m; }

        void compute_dz();

        void forward();
        void backprop();

        void step(double lr, double beta);

        // Gradient clipping
        double get_sq_grad_sum() const;
        void scale_gradients(double scale);

    private:
        void batch_norm_forward();
        void batch_norm_backprop();

        Matrix activation();
};
