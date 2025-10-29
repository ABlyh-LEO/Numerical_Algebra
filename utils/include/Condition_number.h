//
// Created by Yanhong Liu on 2025/10/8.
//

#ifndef NUMERICAL_ALGEBRA_CONDITION_NUMBER_H
#define NUMERICAL_ALGEBRA_CONDITION_NUMBER_H

#include "LU_solver.h"
#include "matrix_utils.h"

template<typename T, int _dim>
T condition_number_oo(const Matrix<T, _dim, _dim> &A) {
    T norm_A = A.oo_norm();
    Matrix<T, _dim, 1> x;
    for (int i = 0; i < _dim; ++i) {
        x[i][0] = 1.0 / _dim;
    }
    PLU_Solver<T, _dim> solver;
    solver.compute(A);
    T norm_A_inv;
    for (;;) {
        auto w = solver.solveT(x);
        auto v = w.sign();
        auto z = solver.solve(v);
        auto z_oo_norm = z.oo_norm();
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
