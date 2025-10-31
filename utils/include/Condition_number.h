//
// Created by Yanhong Liu on 2025/10/8.
//

#ifndef NUMERICAL_ALGEBRA_CONDITION_NUMBER_H
#define NUMERICAL_ALGEBRA_CONDITION_NUMBER_H

#include "LU_solver.h"
#include "matrix_utils.h"

template<typename T>
T condition_number_oo(const Matrix<T> &A) {
    if (A.cols() != A.rows()) {
        std::cerr << "Error(condition_number_oo): Matrix is not square!" << std::endl;
        return -1;
    }
    const int _dim = A.cols();
    T norm_A = A.norm_oo();
    Matrix<T> x(_dim, 1);
    for (int i = 0; i < _dim; ++i) {
        x[i][0] = 1.0 / _dim;
    }
    PLU_Solver<T> solver;
    solver.compute(A);
    T norm_A_inv;
    for (;;) {
        auto w = solver.solveT(x);
        auto v = w.sign();
        auto z = solver.solve(v);
        auto z_oo_norm = z.norm_oo();
        if (z_oo_norm <= (z.transpose() * x)[0][0]) {
            norm_A_inv = w.norm_1();
            break;
        } else {
            for (int i = 0; i < _dim; ++i) {
                x[i][0] = 0;
            }
            for (int i = 0; i < _dim; ++i) {
                if (std::abs(z[i][0]) == z_oo_norm) {
                    x[i][0] = 1;
                    break;
                }
            }
        }
    }
    return norm_A * norm_A_inv;
}


#endif //NUMERICAL_ALGEBRA_CONDITION_NUMBER_H
