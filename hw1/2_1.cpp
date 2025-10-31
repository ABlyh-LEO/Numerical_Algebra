//
// Created by Yanhong Liu on 2025/9/30.
//
#include "Cholesky_solver.h"
#include <random>

#define TYPE float

int main() {
    Matrix<TYPE> A(100, 100);
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            if (i == j) {
                A[i][j] = 10;
            } else if (i == j + 1) {
                A[i][j] = 1;
            } else if (i == j - 1) {
                A[i][j] = 1;
            }
        }
    }

    Matrix<TYPE> b(100, 1);
//    for (int i = 1; i < 99; ++i) {
//        b[i][0] = 12;
//    }
//    b[0][0] = 11;
//    b[99][0] = 11;

    for (int i = 0; i < 100; ++i) {
        b[i][0] = std::rand() / 100.0f;
    }

    Cholesky_solver<TYPE> solver1;
    solver1.compute(A);
    Matrix<TYPE> x1 = solver1.solve(b);
    x1.transpose().print();

    Better_Cholesky_solver<TYPE> solver2;
    solver2.compute(A);
    Matrix<TYPE> x2 = solver2.solve(b);
    x2.transpose().print();

    return 0;
}