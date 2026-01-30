// matrix.cpp

#include "Matrix.hpp"
#include <iostream>
#include <Accelerate/Accelerate.h>

Matrix::Matrix() : rows_(0), cols_(0), data(0) {}

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data(rows * cols) {}

Matrix::Matrix(size_t rows, size_t cols, std::vector<double> data) : rows_(rows), cols_(cols), data(std::move(data)) {}

Matrix::Matrix(Matrix&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data(std::move(other.data))
{
    other.rows_ = 0;
    other.cols_ = 0;
}

double Matrix::get(size_t row, size_t col) const { return data[row * this->cols_ + col]; }
void Matrix::set(size_t row, size_t col, double value) { data[row * this->cols_ + col] = value; }

void Matrix::fill(double value) { std::fill(data.begin(), data.end(), value); }

size_t Matrix::rows() const { return rows_; }
size_t Matrix::cols() const { return cols_; }


Matrix& Matrix::operator+=(const Matrix& other) 
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    vDSP_vaddD(data.data(), 1, other.data.data(), 1, data.data(), 1, rows_ * cols_);
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) 
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    vDSP_vsubD(other.data.data(), 1, data.data(), 1, data.data(), 1, rows_ * cols_);
    return *this;
}

Matrix& Matrix::operator*=(double scalar) 
{
    vDSP_vsmulD(data.data(), 1, &scalar, data.data(), 1, rows_ * cols_);
    return *this;
}

Matrix& Matrix::operator/=(double scalar) 
{
    if (scalar == 0.0)
    {
        throw std::invalid_argument("Error: Division by zero");
    }
    double inv_scalar = 1.0 / scalar;
    vDSP_vsmulD(data.data(), 1, &inv_scalar, data.data(), 1, rows_ * cols_);
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other)
{
    if (cols_ != other.rows_)
    {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result = *this * other;
    *this = std::move(result);
    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows_, cols_);
    vDSP_vaddD(data.data(), 1, other.data.data(), 1, result.data.data(), 1, rows_ * cols_);
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows_, cols_);
    vDSP_vsubD(other.data.data(), 1, data.data(), 1, result.data.data(), 1, rows_ * cols_);
    return result;
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this != &other)
    {
        rows_ = other.rows_;
        cols_ = other.cols_;
    data = other.data;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
    if (this != &other)
    {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data = std::move(other.data);
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(rows_, cols_);
    vDSP_vsmulD(data.data(), 1, &scalar, result.data.data(), 1, rows_ * cols_);
    return result;
}

Matrix Matrix::operator/(double scalar) const
{
    if (scalar == 0.0)
    {
        throw std::invalid_argument("Error: Division by zero");
    }
    Matrix result(rows_, cols_);
    double inv_scalar = 1.0 / scalar;
    vDSP_vsmulD(data.data(), 1, &inv_scalar, result.data.data(), 1, rows_ * cols_);
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (cols_ != other.rows_)
    {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    
    // C = alpha * A * B + beta * C
    // result = 1.0 * this * other + 0.0 * result
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(rows_),           // M: rows of A and C
                static_cast<int>(other.cols_),     // N: cols of B and C
                static_cast<int>(cols_),           // K: cols of A, rows of B
                1.0,                               // alpha
                data.data(),                       // A
                static_cast<int>(cols_),           // lda: leading dimension of A
                other.data.data(),                 // B
                static_cast<int>(other.cols_),     // ldb: leading dimension of B
                0.0,                               // beta
                result.data.data(),                // C
                static_cast<int>(other.cols_));    // ldc: leading dimension of C
    
    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const
{
    Matrix result(rows_, cols_);
    vDSP_vmulD(data.data(), 1, other.data.data(), 1, result.data.data(), 1, rows_ * cols_);
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix trans = Matrix(cols_, rows_);

    vDSP_mtransD(data.data(), 1, trans.data.data(), 1, rows_, cols_);

    return trans;
}

// Activation functions

Matrix Matrix::relu() const
{
    Matrix relu = Matrix(rows_, cols_);

    for (size_t i = 0; i < rows_ * cols_; i++)
    {
        relu.data[i] = data[i] > 0.0 ? data[i] : 0.0;
    }

    return relu;
}

Matrix Matrix::drelu() const
{
    Matrix drelu = Matrix(rows_, cols_);

    for (size_t i = 0; i < rows_ * cols_; i++)
    {
        drelu.data[i] = data[i] > 0 ? 1 : 0;
    }

    return drelu;
}

Matrix Matrix::softmax() const
{
    Matrix softmax(rows_, cols_);

    for (size_t c = 0; c < cols_; ++c)
    {
        // 1) max nella colonna c
        double max_val = data[c];
        for (size_t r = 1; r < rows_; ++r)
        {
            double v = data[r * cols_ + c];
            if (v > max_val) max_val = v;
        }

        // 2) exp(x - max) e somma
        double sum = 0.0;
        for (size_t r = 0; r < rows_; ++r)
        {
            double e = std::exp(data[r * cols_ + c] - max_val);
            softmax.data[r * cols_ + c] = e;
            sum += e;
        }

        // 3) normalizzazione
        if (sum != 0.0)
        {
            for (size_t r = 0; r < rows_; ++r)
            {
                softmax.data[r * cols_ + c] /= sum;
            }
        }
    }

    return softmax;
}

// Operations

void Matrix::add_col_vector(const Matrix& b)
{
    if (b.cols() != 1)
    {
        throw std::invalid_argument("Error: add_col_vector requires a column vector (1 column)");
    }
    if (b.rows() != rows_)
    {
        throw std::invalid_argument("Error: Vector rows must match matrix rows for add_col_vector");
    }
    
    // Add vector b to each column of the matrix
    for (size_t c = 0; c < cols_; c++)
    {
        vDSP_vaddD(&data[c], cols_, b.data.data(), 0, &data[c], cols_, rows_);
    }
}

void Matrix::mul_col_vector(const Matrix& b)
{
    if (b.cols() != 1)
    {
        throw std::invalid_argument("Error: mul_col_vector requires a column vector (1 column)");
    }
    if (b.rows() != rows_)
    {
        throw std::invalid_argument("Error: Vector rows must match matrix rows for mul_col_vector");
    }
    
    // Multiply vector b with each column of the matrix
    // For each column c: data[:,c] *= b
    for (size_t c = 0; c < cols_; c++)
    {
        vDSP_vmulD(&data[c], cols_, b.data.data(), 0, &data[c], cols_, rows_);
    }
}

void Matrix::multiply_col(size_t col_idx, double scalar)
{
    if (col_idx >= cols_)
    {
        throw std::invalid_argument("Error: Column index out of bounds");
    }
    
    vDSP_vsmulD(&data[col_idx], cols_, &scalar, &data[col_idx], cols_, rows_);
}

Matrix Matrix::normalize(const Matrix& mean, const Matrix& var) const {
    const double epsilon = 1e-5;
    Matrix result(rows_, cols_);
    result.data = data; // Start copy

    for (size_t r = 0; r < rows_; r++) {
        double m = mean.get(r, 0);
        double v = var.get(r, 0);
        
        double inv_std = 1.0 / std::sqrt(v + epsilon);
        double neg_m = -m;
        
        // dst = (src - m) * inv_std = (src + neg_m) * inv_std
        vDSP_vsaddD(&result.data[r * cols_], 1, &neg_m, &result.data[r * cols_], 1, cols_);
        vDSP_vsmulD(&result.data[r * cols_], 1, &inv_std, &result.data[r * cols_], 1, cols_);
    }
    return result;
}

Matrix Matrix::normalize_derivative(const Matrix& mean, const Matrix& var, const Matrix& z_norm) const {
    const double epsilon = 1e-5;
    Matrix dZ(rows_, cols_);
    size_t B = cols_;
    double inv_B = 1.0 / B;
    double m_val = static_cast<double>(B);

    for (size_t r = 0; r < rows_; r++) {
        double v = var.get(r, 0);
        double std = std::sqrt(v + epsilon);
        double inv_std = 1.0 / std;
        double scale = inv_std * inv_B; // 1/(m*sigma)

        // data pointers for this row
        const double* dy_row = &data[r * cols_]; 
        const double* x_hat_row = &z_norm.data[r * cols_];
        double* dx_row = &dZ.data[r * cols_];

        // 1. sum_dy = sum(dy)
        double sum_dy = 0.0;
        vDSP_sveD(dy_row, 1, &sum_dy, cols_);

        // 2. sum_dy_xhat = sum(dy * x_hat)
        double sum_dy_xhat = 0.0;
        vDSP_dotprD(dy_row, 1, x_hat_row, 1, &sum_dy_xhat, cols_);

        // 3. Compute dx
        // dx = m * dy
        vDSP_vsmulD(dy_row, 1, &m_val, dx_row, 1, cols_);

        // dx = dx - sum_dy
        double neg_sum_dy = -sum_dy;
        vDSP_vsaddD(dx_row, 1, &neg_sum_dy, dx_row, 1, cols_);

        // dx = dx - x_hat * sum_dy_xhat
        double neg_sum_dy_xhat = -sum_dy_xhat;
        vDSP_vsmaD(x_hat_row, 1, &neg_sum_dy_xhat, dx_row, 1, dx_row, 1, cols_);

        // dx = dx * scale
        vDSP_vsmulD(dx_row, 1, &scale, dx_row, 1, cols_);
    }
    return dZ;
}

Matrix Matrix::sum_columns() const
{
    Matrix result(rows_, 1);

    // Sum each row across all columns
    for (size_t r = 0; r < rows_; ++r)
    {
        double sum = 0.0;
        vDSP_sveD(&data[r * cols_], 1, &sum, cols_);
        result.set(r, 0, sum);
    }

    return result;
}

double Matrix::sum_of_squares() const
{
    double result = 0.0;
    vDSP_svesqD(data.data(), 1, &result, rows_ * cols_);
    return result;
}

size_t Matrix::argmax_col(size_t col_idx) const
{
    size_t max_idx = 0;
    double max_val = get(0, col_idx);

    for (size_t i = 1; i < rows_; i++)
    {
        double val = get(i, col_idx);
        if (val > max_val)
        {
            max_idx = i;
            max_val = val;
        }
    }
    return max_idx;
}

Matrix Matrix::mean() const {
    Matrix m(rows_, 1);
    for (size_t r = 0; r < rows_; r++) {
        double row_mean = 0.0;
        vDSP_meanvD(&data[r * cols_], 1, &row_mean, cols_);
        m.set(r, 0, row_mean);
    }
    return m;
}

Matrix Matrix::variance(const Matrix& mean) const {
    Matrix v(rows_, 1);
    for (size_t r = 0; r < rows_; r++) {
        double row_mean = mean.get(r, 0);
        double sq_sum = 0.0;
        vDSP_svesqD(&data[r * cols_], 1, &sq_sum, cols_);
        double row_var = (sq_sum / cols_) - (row_mean * row_mean);
        v.set(r, 0, row_var);
    }
    return v;
}

void Matrix::print() const
{
    for (size_t r = 0; r < rows_; r++)
    {
        for (size_t c = 0; c < cols_; c++)
        {
            std::cout << data[r * cols_ + c] << " ";
        }
        std::cout << std::endl;
    }
}