//
// Created by Yanhong Liu on 2025/9/30.
//

#ifndef NUMERICAL_ALGEBRA_CHOLESKY_SOLVER_H
#define NUMERICAL_ALGEBRA_CHOLESKY_SOLVER_H

#include <valarray>
#include "matrix_utils.h"

template<typename T>
class Cholesky_solver {
private:
    Matrix<T> L;
    int _dim;
public:
    Cholesky_solver() = default;

    ~Cholesky_solver() = default;

    void compute(const Matrix<T> &A) {
        if (A.cols() != A.rows()) {
            std::cerr << "Error(Cholesky_solver): Matrix is not square!" << std::endl;
            return;
        }
        _dim = A.cols();
        L = Matrix<T>(_dim, _dim);
        for (int i = 0; i < _dim; ++i) {
            for (int j = 0; j <= i; ++j) {
                T sum = 0;
                for (int k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j) {
                    if (A[i][i] - sum <= 0) {
                        std::cerr << "Error(Cholesky_solver): Trying to compute the square root of a negative number!" << std::endl;
                        return;
                    }
                    L[i][j] = std::sqrt(A[i][i] - sum);
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
    }

    Matrix<T> solve(const Matrix<T> &b) {
        if (_dim <= 0) {
            std::cerr << "Error(Cholesky_solver): Matrix is not computed!" << std::endl;
            return Matrix<T>();
        }
        Matrix<T> y(_dim, 1), x(_dim, 1);
        for (int i = 0; i < _dim; ++i) {
            y[i][0] = b[i][0];
            for (int j = 0; j < i; ++j) {
                y[i][0] -= L[i][j] * y[j][0];
            }
            y[i][0] /= L[i][i];
        }
        for (int i = _dim - 1; i >= 0; --i) {
            x[i][0] = y[i][0];
            for (int j = i + 1; j < _dim; ++j) {
                x[i][0] -= L[j][i] * x[j][0];
            }
            x[i][0] /= L[i][i];
        }
        return x;
    }
};

template<typename T>
class Better_Cholesky_solver { //LDLT
private:
    Matrix<T> L;
    std::vector<T> v{};
    int _dim;
public:
    Better_Cholesky_solver() = default;

    ~Better_Cholesky_solver() = default;

    void compute(const Matrix<T> &A) {
        if (A.cols() != A.rows()) {
            std::cerr << "Error(Better_Cholesky_solver): Matrix is not square!" << std::endl;
            return;
        }
        _dim = A.cols();
        v.resize(_dim);
        L = Matrix<T>(_dim, _dim);
        for (int i = 0; i < _dim; ++i) {
            for (int j = 0; j <= i; ++j) {
                T sum = 0;
                for (int k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k] * v[k];
                }
                if (i == j) {
                    if (A[i][i] - sum <= 0) std::cerr << "Warning(Better_Cholesky_solver): Matrix D has negative elements!" << std::endl;
                    v[i] = A[i][i] - sum;
                    L[i][i] = 1;
                } else {
                    L[i][j] = (A[i][j] - sum) / v[j];
                }
            }
        }
    }

    Matrix<T> solve(const Matrix<T> &b) {
        if (_dim <= 0) {
            std::cerr << "Error(Better_Cholesky_solver): Matrix is not computed!" << std::endl;
            return Matrix<T>();
        }
        Matrix<T> y(_dim, 1), z(_dim, 1), x(_dim, 1);
        for (int i = 0; i < _dim; ++i) {
            y[i][0] = b[i][0];
            for (int j = 0; j < i; ++j) {
                y[i][0] -= L[i][j] * y[j][0];
            }
        }
        for (int i = 0; i < _dim; ++i) {
            z[i][0] = y[i][0] / v[i];
        }
        for (int i = _dim - 1; i >= 0; --i) {
            x[i][0] = z[i][0];
            for (int j = i + 1; j < _dim; ++j) {
                x[i][0] -= L[j][i] * x[j][0];
            }
        }
        return x;
    }
};

#endif //NUMERICAL_ALGEBRA_CHOLESKY_SOLVER_H
