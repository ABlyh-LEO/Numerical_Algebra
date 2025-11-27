//
// Created by Yanhong Liu on 2025/11/27.
//

#ifndef NUMERICAL_ALGEBRA_IT_METHOD_H
#define NUMERICAL_ALGEBRA_IT_METHOD_H

#include "matrix_utils.h"
#include "Sparse_utils.h"

template<typename T>
int Jacobi_Method(const Matrix<T> &A, const Matrix<T> &b, Matrix<T> &x, int max_iter = 10000, T tol = 1e-4) {
    for (int n = 1; n <= max_iter; ++n) {
        auto x_last = x;
        for (int i = 0; i < A.rows(); ++i) {
            T sum = 0;
            for (int j = 0; j < A.cols(); ++j) {
                if (j != i) {
                    sum += A[i][j] * x_last[j][0];
                }
            }
            x[i][0] = (b[i][0] - sum) / A[i][i];
        }
        if ((x - x_last).norm_F() < tol) {
            return n;
        }
    }
    return max_iter;
}

template<typename T>
int GS_Method(const Matrix<T> &A, const Matrix<T> &b, Matrix<T> &x, int max_iter = 10000, T tol = 1e-4) {
    for (int n = 1; n <= max_iter; ++n) {
        auto x_last = x;
        for (int i = 0; i < A.rows(); ++i) {
            T sum = 0;
            for (int j = 0; j < A.cols(); ++j) {
                if (j != i) {
                    sum += A[i][j] * x[j][0];
                }
            }
            x[i][0] = (b[i][0] - sum) / A[i][i];
        }
        if ((x - x_last).norm_F() < tol) {
            return n;
        }
    }
    return max_iter;
}

template<typename T>
int GS_Method(const SparseMatrix<T> &A, const Matrix<T> &b, Matrix<T> &x, int max_iter = 10000, T tol = 1e-4) {
    for (int n = 1; n <= max_iter; ++n) {
        auto x_last = x;
        for (int i = 0; i < A.rows_; ++i) {
            T sum = 0;
            T aii;
            for (int j = A.row_ptrs_[i]; j < A.row_ptrs_[i + 1]; ++j) {
                if (A.col_indices_[j] != i) {
                    sum += A.values_[j] * x[A.col_indices_[j]][0];
                } else {
                    aii = A.values_[j];
                }
            }
            x[i][0] = (b[i][0] - sum) / aii;
        }
        if ((x - x_last).norm_F() < tol) {
            return n;
        }
    }
    return max_iter;
}

template<typename T>
int SOR_Method(const Matrix<T> &A, const Matrix<T> &b, Matrix<T> &x, int max_iter = 10000, T tol = 1e-4, T omega = 1.5) {
    for (int n = 1; n <= max_iter; ++n) {
        auto x_last = x;
        for (int i = 0; i < A.rows(); ++i) {
            T sum = 0;
            for (int j = 0; j < A.cols(); ++j) {
                if (j != i) {
                    sum += A[i][j] * x[j][0];
                }
            }
            x[i][0] = (1 - omega) * x[i][0] + omega * (b[i][0] - sum) / A[i][i];
        }
        if ((x - x_last).norm_F() < tol) {
            return n;
        }
    }
    return max_iter;
}

#endif //NUMERICAL_ALGEBRA_IT_METHOD_H
