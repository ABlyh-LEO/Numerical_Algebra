//
// Created by Yanhong Liu on 2025/9/29.
//
#include "matrix_utils.h"
#include "LU_solver.h"

#define TYPE long double

int main() {
    Matrix<TYPE, 86, 86> A;
    for (int i = 0; i < 86; ++i) {
        for (int j = 0; j < 86; ++j) {
            if (i == j) {
                A[i][j] = 6;
            } else if (i == j + 1) {
                A[i][j] = 8;
            } else if (i == j - 1) {
                A[i][j] = 1;
            }
        }
    }

    Matrix<TYPE, 86, 1> b;
    b[0][0] = 7;
    for (int i = 1; i < 85; ++i) {
        b[i][0] = 15;
    }
    b[85][0] = 14;

    LU_Solver<TYPE, 86> solver1;
    solver1.compute(A);
    Matrix<TYPE, 86, 1> x1 = solver1.solve(b);
    x1.transpose().print();

    PLU_Solver<TYPE, 86> solver2;
    solver2.compute(A);
    Matrix<TYPE, 86, 1> x2 = solver2.solve(b);
    x2.transpose().print();

    return 0;
}