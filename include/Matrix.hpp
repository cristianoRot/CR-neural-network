// matrix.hpp

#pragma once
#include <iostream>
#include <vector>

class Matrix 
{
    private:
        size_t row, col;
        std::vector<double> data;

    public:
        Matrix();
        Matrix(size_t row, size_t col);
        Matrix(size_t row, size_t col, std::vector<double> data);

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
        Matrix& operator*=(const Matrix& other);
        Matrix& operator=(const Matrix& other);

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(double scalar) const;
        Matrix operator*(const Matrix& other) const;

        Matrix hadamard(const Matrix& other) const;
        Matrix transpose() const;


        // Activation functions
        Matrix relu() const;
        Matrix drelu() const;

        Matrix softmax() const;

        void print() const;
};
