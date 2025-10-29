//
// Created by Yanhong Liu on 2025/9/30.
//
#include "Cholesky_solver.h"

#define TYPE long double

int main() {
    Matrix<TYPE, 40, 40> A;
    for (int i = 0; i < 40; ++i) {
        for (int j = 0; j < 40; ++j) {
            A[i][j] = 1.0 / (i + j + 1);
        }
    }
    Matrix<TYPE, 40, 1> b;
    for (int i = 0; i < 40; ++i) {
        for (int j = 0; j < 40; ++j) {
            b[i][0] += 1.0 / (i + j + 1);
        }
    }

    Cholesky_solver<TYPE, 40> solver1;
    solver1.compute(A);
    Matrix<TYPE, 40, 1> x1 = solver1.solve(b);
    x1.transpose().print();

    Better_Cholesky_solver<TYPE, 40> solver2;
    solver2.compute(A);
    Matrix<TYPE, 40, 1> x2 = solver2.solve(b);
    x2.transpose().print();

    return 0;
}