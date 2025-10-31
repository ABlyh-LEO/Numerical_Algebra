//
// Created by Yanhong Liu on 2025/9/30.
//

#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <cstring>
#include <stdexcept>
#include <iostream>
#include <valarray>
#include <vector>


template<typename T>
class Matrix {
protected:
    int rows_, cols_;
    std::vector<T> data_;
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, T(0)) {}

    Matrix() {
        rows_ = cols_ = -1;
    }

    void set_shape(int rows, int cols) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows * cols, T(0));
    }

    explicit Matrix(int rows, int cols, const std::vector<T> &data) : Matrix(rows, cols) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument("Data size does not match matrix dimensions");
        }
        std::copy(data.begin(), data.end(), data_.begin());
    }

    Matrix(const Matrix<T> &mat) : Matrix(mat.rows_, mat.cols_) {
        std::copy(mat.data_.begin(), mat.data_.end(), data_.begin());
    }

    ~Matrix() = default;

    [[nodiscard]] int rows() const { return rows_; }

    [[nodiscard]] int cols() const { return cols_; }

    T *operator[](const int &row) { return &this->data_[row * cols_]; }

    const T *operator[](const int &row) const { return &this->data_[row * cols_]; }

    Matrix<T> &operator=(const Matrix<T> &mat) {
        if (this == &mat) return *this;
        this->set_shape(mat.rows_, mat.cols_);
        std::copy(mat.data_.begin(), mat.data_.end(), data_.begin());
        return *this;
    }

    T norm_1() const {
        T max_sum = 0;
        for (int j = 0; j < cols_; ++j) {
            T col_sum = 0;
            for (int i = 0; i < rows_; ++i) {
                col_sum += std::abs(this->data_[i * cols_ + j]);
            }
            if (col_sum > max_sum) {
                max_sum = col_sum;
            }
        }
        return max_sum;
    }

    T norm_oo() const {
        T max_sum = 0;
        for (int i = 0; i < rows_; ++i) {
            T row_sum = 0;
            for (int j = 0; j < cols_; ++j) {
                row_sum += std::abs(this->data_[i * cols_ + j]);
            }
            if (row_sum > max_sum) {
                max_sum = row_sum;
            }
        }
        return max_sum;
    }

    T norm_F() const {
        T sum = 0;
        for (int i = 0; i < rows_ * cols_; ++i) {
            sum += this->data_[i] * this->data_[i];
        }
        return std::sqrt(sum);
    }

    Matrix<T> &operator*=(const T &val) {
        for (int i = 0; i < rows_ * cols_; ++i) {
            this->data_[i] *= val;
        }
        return *this;
    }

    Matrix<T> &operator/=(const T &val) {
        if (val == 0) throw std::invalid_argument("Division by zero");
        for (int i = 0; i < rows_ * cols_; ++i) {
            this->data_[i] /= val;
        }
        return *this;
    }

    Matrix<T> &operator+=(const Matrix<T> &mat) {
        if (rows_ != mat.rows_ || cols_ != mat.cols_) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }
        for (int i = 0; i < rows_ * cols_; ++i) {
            this->data_[i] += mat.data_[i];
        }
        return *this;
    }

    Matrix<T> &operator-=(const Matrix<T> &mat) {
        if (rows_ != mat.rows_ || cols_ != mat.cols_) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }
        for (int i = 0; i < rows_ * cols_; ++i) {
            this->data_[i] -= mat.data_[i];
        }
        return *this;
    }

    Matrix<T> operator*(const T &val) const {
        Matrix<T> res(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            res.data_[i] = this->data_[i] * val;
        }
        return res;
    }

    friend Matrix<T> operator*(const T &val, const Matrix<T> &mat) {
        Matrix<T> res(mat.rows_, mat.cols_);
        for (int i = 0; i < mat.rows_ * mat.cols_; ++i) {
            res.data_[i] = mat.data_[i] * val;
        }
        return res;
    }

    Matrix<T> operator+(const Matrix<T> &mat) const {
        if (rows_ != mat.rows_ || cols_ != mat.cols_) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }
        Matrix<T> res(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            res.data_[i] = this->data_[i] + mat.data_[i];
        }
        return res;
    }

    Matrix<T> operator-(const Matrix<T> &mat) const {
        if (rows_ != mat.rows_ || cols_ != mat.cols_) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }
        Matrix<T> res(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            res.data_[i] = this->data_[i] - mat.data_[i];
        }
        return res;
    }

    Matrix<T> operator*(const Matrix<T> &mat) const {
        if (cols_ != mat.rows_) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        Matrix<T> res(rows_, mat.cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < mat.cols_; ++j) {
                res[i][j] = 0;
                for (int k = 0; k < cols_; ++k) {
                    res[i][j] += this->data_[i * cols_ + k] * mat.data_[k * mat.cols_ + j];
                }
            }
        }
        return res;
    }

    Matrix<T> transpose() const {
        Matrix<T> res(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                res[j][i] = this->data_[i * cols_ + j];
            }
        }
        return res;
    }

    Matrix<T> sign() const {
        Matrix<T> res(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i) {
            if (this->data_[i] > 0) res.data_[i] = 1;
            else if (this->data_[i] < 0) res.data_[i] = -1;
            else res.data_[i] = 0;
        }
        return res;
    }

    void print() const {
        std::cout << "[ " << std::endl;
        for (int i = 0; i < rows_; ++i) {
            std::cout << "  [ ";
            for (int j = 0; j < cols_; ++j) {
                std::cout << this->data_[i * cols_ + j] << ((j == cols_ - 1) ? "" : ", ");
            }
            std::cout << " ]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
};

//template<typename T, int _rows, int _cols>
//class Matrix {
//protected:
//    int rows_, cols_;
//    T data_[_rows * _cols]{};
//public:
//    Matrix() {
//        rows_ = _rows;
//        cols_ = _cols;
//        memset(this->data_, 0, _rows * _cols * sizeof(T));
//    }
//
//    explicit Matrix(T data[_rows * _cols]) : Matrix() {
//        memcpy(this->data_, data, _rows * _cols * sizeof(T));
//    }
//
//    Matrix(const Matrix<T, _rows, _cols> &mat) : Matrix() {
//        memcpy(this->data_, mat.data_, _rows * _cols * sizeof(T));
//    }
//
//    ~Matrix() = default;
//
//    int rows() { return _rows; }
//
//    int cols() { return _cols; }
//
//    T *operator[](const int &row) { return &this->data_[row * _cols]; }
//
//    const T *operator[](const int &row) const { return &this->data_[row * _cols]; }
//
//    Matrix<T, _rows, _cols> &operator=(const Matrix<T, _rows, _cols> &mat) {
//        if (this == &mat) return *this;
//        memcpy(this->data_, mat.data_, _rows * _cols * sizeof(T));
//        return *this;
//    }
//
//    Matrix<T, _rows, _cols> &operator+=(const Matrix<T, _rows, _cols> &mat) {
//        for (int i = 0; i < _rows * _cols; ++i) {
//            this->data_[i] += mat.data_[i];
//        }
//        return *this;
//    }
//
//    Matrix<T, _rows, _cols> &operator-=(const Matrix<T, _rows, _cols> &mat) {
//        for (int i = 0; i < _rows * _cols; ++i) {
//            this->data_[i] -= mat.data_[i];
//        }
//        return *this;
//    }
//
//    Matrix<T, _rows, _cols> &operator*=(const T &val) {
//        for (int i = 0; i < _rows * _cols; ++i) {
//            this->data_[i] *= val;
//        }
//        return *this;
//    }
//
//    Matrix<T, _rows, _cols> &operator/=(const T &val) {
//        if (val == 0) throw std::invalid_argument("Division by zero");
//        for (int i = 0; i < _rows * _cols; ++i) {
//            this->data_[i] /= val;
//        }
//        return *this;
//    }
//
//    Matrix<T, _rows, _cols> operator+(const Matrix<T, _rows, _cols> &mat) {
//        Matrix<T, _rows, _cols> res;
//        for (int i = 0; i < _rows * _cols; ++i) {
//            res.data_[i] = this->data_[i] + mat.data_[i];
//        }
//        return res;
//    }
//
//    Matrix<T, _rows, _cols> operator-(const Matrix<T, _rows, _cols> &mat) {
//        Matrix<T, _rows, _cols> res;
//        for (int i = 0; i < _rows * _cols; ++i) {
//            res.data_[i] = this->data_[i] - mat.data_[i];
//        }
//        return res;
//    }
//
//    Matrix<T, _rows, _cols> operator*(const T &val) {
//        Matrix<T, _rows, _cols> res;
//        for (int i = 0; i < _rows * _cols; ++i) {
//            res.data_[i] = this->data_[i] * val;
//        }
//        return res;
//    }
//
//    friend Matrix<T, _rows, _cols> operator*(const T &val,
//                                             const Matrix<T, _rows, _cols> &mat) {
//        Matrix<T, _rows, _cols> res;
//        for (int i = 0; i < _rows * _cols; ++i) {
//            res.data_[i] = mat.data_[i] * val;
//        }
//        return res;
//    }
//
//    template<int cols_>
//    friend Matrix<T, _rows, cols_> operator*(const Matrix<T, _rows, _cols> &a,
//                                             const Matrix<T, _cols, cols_> &b) {
//        Matrix<T, _rows, cols_> res;
//        for (int i = 0; i < _rows; ++i) {
//            for (int j = 0; j < cols_; ++j) {
//                res[i][j] = 0;
//                for (int k = 0; k < _cols; ++k) {
//                    res[i][j] += a[i][k] * b[k][j];
//                }
//            }
//        }
//        return res;
//    }
//
//
//    Matrix<T, 1, _cols> row(int r) {
//        if (r < 0 || r >= _rows) throw std::out_of_range("Row index out of range");
//        Matrix<T, 1, _cols> res;
//        memcpy(res[0], &this->data_[r * _cols], _cols * sizeof(T));
//        return res;
//    }
//
//    Matrix<T, _rows, 1> col(int c) {
//        if (c < 0 || c >= _cols) throw std::out_of_range("Column index out of range");
//        Matrix<T, _rows, 1> res;
//        for (int i = 0; i < _rows; ++i) {
//            res[i][0] = this->data_[i * _cols + c];
//        }
//        return res;
//    }
//
//    void add_to_row(int r, const Matrix<T, 1, _cols> &vec) {
//        if (r < 0 || r >= _rows) throw std::out_of_range("Row index out of range");
//        for (int j = 0; j < _cols; ++j) {
//            this->data_[r * _cols + j] += vec[0][j];
//        }
//    }
//
//    void add_to_col(int c, const Matrix<T, _rows, 1> &vec) {
//        if (c < 0 || c >= _cols) throw std::out_of_range("Column index out of range");
//        for (int i = 0; i < _rows; ++i) {
//            this->data_[i * _cols + c] += vec[i][0];
//        }
//    }
//
//    Matrix<T, _cols, _rows> transpose() const {
//        Matrix<T, _cols, _rows> res;
//        for (int i = 0; i < _rows; ++i) {
//            for (int j = 0; j < _cols; ++j) {
//                res[j][i] = this->data_[i * _cols + j];
//            }
//        }
//        return res;
//    }
//
//    Matrix<T, _rows, _cols> sign() const {
//        Matrix<T, _rows, _cols> res;
//        for (int i = 0; i < _rows * _cols; ++i) {
//            if (this->data_[i] > 0) res.data_[i] = 1;
//            else if (this->data_[i] < 0) res.data_[i] = -1;
//            else res.data_[i] = 0;
//        }
//        return res;
//    }
//
//    T oo_norm() const {
//        T max_sum = 0;
//        for (int i = 0; i < _rows; ++i) {
//            T row_sum = 0;
//            for (int j = 0; j < _cols; ++j) {
//                row_sum += std::abs(this->data_[i * _cols + j]);
//            }
//            if (row_sum > max_sum) {
//                max_sum = row_sum;
//            }
//        }
//        return max_sum;
//    }
//
//    T norm_1() const {
//        T max_sum = 0;
//        for (int j = 0; j < _cols; ++j) {
//            T col_sum = 0;
//            for (int i = 0; i < _rows; ++i) {
//                col_sum += std::abs(this->data_[i * _cols + j]);
//            }
//            if (col_sum > max_sum) {
//                max_sum = col_sum;
//            }
//        }
//        return max_sum;
//    }
//
//    T F_norm() const {
//        T sum = 0;
//        for (int i = 0; i < _rows * _cols; ++i) {
//            sum += this->data_[i] * this->data_[i];
//        }
//        return std::sqrt(sum);
//    }
//
//    void print() const {
//        std::cout << "[ ";
//        for (int i = 0; i < _rows; ++i) {
//            std::cout << "[ ";
//            for (int j = 0; j < _cols; ++j) {
//                std::cout << this->data_[i * _cols + j] << ((j == _cols - 1) ? "" : ", ");
//            }
//            std::cout << " ]" << std::endl;
//        }
//        std::cout << " ]" << std::endl;
//    }
//};

#endif
