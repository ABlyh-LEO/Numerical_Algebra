//
// Created by Yanhong Liu on 2025/10/30.
//
#include "matrix_utils.h"
#include "LU_solver.h"
#include "QR_solver.h"

#define TYPE long double

int main() {
    Matrix<TYPE> A(86, 86);
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

    Matrix<TYPE> b(86, 1);
    b[0][0] = 7;
    for (int i = 1; i < 85; ++i) {
        b[i][0] = 15;
    }
    b[85][0] = 14;

    QR_Solver<TYPE> solver1;
    solver1.compute(A);
    Matrix<TYPE> x1 = solver1.solve(b);
    x1.transpose().print();

    return 0;
}