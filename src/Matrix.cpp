// matrix.cpp

#include "Matrix.hpp"

Matrix::Matrix() : row(0), col(0), data(0) {}

Matrix::Matrix(size_t row, size_t col) : row(row), col(col), data(row * col) {}

Matrix::Matrix(size_t row, size_t col, std::vector<double> data) : row(row), col(col), data(data) {}

double Matrix::get(size_t row, size_t col) const { return data[row * this->col + col]; }
void Matrix::set(size_t row, size_t col, double value) { data[row * this->col + col] = value; }

void Matrix::fill(double value) { std::fill(data.begin(), data.end(), value); }

size_t Matrix::rows() const { return row; }
size_t Matrix::cols() const { return col; }


Matrix& Matrix::operator+=(const Matrix& other) 
{
    *this = *this + other;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) 
{
    *this = *this - other;
    return *this;
}

Matrix& Matrix::operator*=(double scalar) 
{
    *this = *this * scalar;
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other)
{
    *this = *this * other;
    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const
{
    if (row != other.row || col != other.col)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(row, col);
    for (size_t i = 0; i < row * col; i++)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    if (row != other.row || col != other.col)
    {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(row, col);
    for (size_t i = 0; i < row * col; i++)
    {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Matrix& Matrix::operator=(const Matrix& other)
{
    row = other.row;
    col = other.col;
    data = other.data;
    return *this;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(row, col);
    for (size_t i = 0; i < row * col; i++)
    {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (col != other.row)
    {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(row, other.col);
    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < other.col; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < col; k++)
            {
                sum += data[i * col + k] * other.data[k * other.col + j];
            }
            result.data[i * other.col + j] = sum;
        }
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const
{
    Matrix result(row, col);
    for (size_t i = 0; i < row * col; i++)
    {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix trans = Matrix(col, row);

    for (int r = 0; r < row; ++r) 
    {
        for (int c = 0; c < col; ++c) 
        {
            trans.data[c * row + r] = data[r * col + c];
        }
    }

    return trans;
}

// Activation functions

Matrix Matrix::relu() const
{
    Matrix relu = Matrix(row, col);

    for (size_t i = 0; i < row * col; i++)
    {
        if (data[i] < 0) relu.data[i] = 0;
        else relu.data[i] = data[i];
    }

    return relu;
}

Matrix Matrix::drelu() const
{
    Matrix drelu = Matrix(row, col);

    for (size_t i = 0; i < row * col; i++)
    {
        drelu.data[i] = data[i] > 0 ? 1 : 0;
    }

    return drelu;
}

Matrix Matrix::softmax() const
{
    Matrix softmax(row, col);

    for (size_t c = 0; c < col; ++c)
    {
        // 1) max nella colonna c
        double max_val = data[c];
        for (size_t r = 1; r < row; ++r)
        {
            double v = data[r * col + c];
            if (v > max_val) max_val = v;
        }

        // 2) exp(x - max) e somma
        double sum = 0.0;
        for (size_t r = 0; r < row; ++r)
        {
            double e = std::exp(data[r * col + c] - max_val);
            softmax.data[r * col + c] = e;
            sum += e;
        }

        // 3) normalizzazione
        if (sum != 0.0)
        {
            for (size_t r = 0; r < row; ++r)
            {
                softmax.data[r * col + c] /= sum;
            }
        }
    }

    return softmax;
}

void Matrix::print() const
{
    for (size_t r = 0; r < row; r++)
    {
        for (size_t c = 0; c < col; c++)
        {
            std::cout << data[r * col + c] << " ";
        }
        std::cout << std::endl;
    }
}