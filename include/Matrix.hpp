// matrix.hpp

#pragma once
#include <iostream>
#include <vector>

class Matrix 
{
    private:
        size_t rows_, cols_;
        std::vector<double> data;

    public:
        Matrix();
        Matrix(size_t rows, size_t cols);
        Matrix(size_t rows, size_t cols, std::vector<double> data);
        Matrix(Matrix&& other) noexcept;
        Matrix(const Matrix& other) = default;

        double get(size_t row, size_t col) const;
        void set(size_t row, size_t col, double value);

        void fill(double value);

        size_t rows() const;
        size_t cols() const;
        
        const std::vector<double>& get_data() const { return data; }
        void set_data(const std::vector<double>& new_data) { data = new_data; }

        Matrix& operator+=(const Matrix& other);
        Matrix& operator-=(const Matrix& other);
        Matrix& operator*=(double scalar);
        Matrix& operator/=(double scalar);
        Matrix& operator*=(const Matrix& other);
        Matrix& operator=(const Matrix& other);
        Matrix& operator=(Matrix&& other) noexcept;

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(double scalar) const;
        Matrix operator/(double scalar) const;
        Matrix operator*(const Matrix& other) const;

        Matrix hadamard(const Matrix& other) const;
        Matrix transpose() const;

        // Activation functions
        Matrix relu() const;
        Matrix drelu() const;

        Matrix softmax() const;

        // In-place operations
        void add_col_vector(const Matrix& b);
        Matrix sum_columns() const;
        size_t argmax_col(size_t col_idx) const;

        void print() const;
};
