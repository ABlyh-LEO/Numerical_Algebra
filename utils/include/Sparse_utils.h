//
// Created by Yanhong Liu on 2025/11/27.
//

#ifndef NUMERICAL_ALGEBRA_SPARSE_UTILS_H
#define NUMERICAL_ALGEBRA_SPARSE_UTILS_H

#include <cstring>
#include <stdexcept>
#include <iostream>
#include <valarray>
#include <vector>
#include <tuple>
#include <matrix_utils.h>

template<typename T>
class SparseMatrix {
public:
    int rows_;
    int cols_;
    std::vector<T> values_;
    std::vector<int> col_indices_;
    std::vector<int> row_ptrs_;

    explicit SparseMatrix(const Matrix<T> &A) {
        rows_ = A.rows();
        cols_ = A.cols();
        row_ptrs_.resize(rows_ + 1, 0);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (A[i][j] != 0) {
                    values_.push_back(A[i][j]);
                    col_indices_.push_back(j);
                    row_ptrs_[i + 1]++;
                }
            }
        }
        for (int i = 1; i <= rows_; ++i) {
            row_ptrs_[i] += row_ptrs_[i - 1];
        }
    }
};

template<typename T>
Matrix<T> operator*(const SparseMatrix<T> &A, const Matrix<T> &x) {
    if (A.cols_ != x.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    Matrix<T> result(A.rows_, x.cols());
    for (int i = 0; i < A.rows_; ++i) {
        for (int j = A.row_ptrs_[i]; j < A.row_ptrs_[i + 1]; ++j) {
            result[i][0] += A.values_[j] * x[A.col_indices_[j]][0];
        }
    }
    return result;
}

#endif //NUMERICAL_ALGEBRA_SPARSE_UTILS_H
