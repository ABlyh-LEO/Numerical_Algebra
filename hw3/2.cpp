//
// Created by Yanhong Liu on 2025/10/31.
//
#include "matrix_utils.h"
#include "QR_solver.h"
#include <iostream>

#define TYPE double

int main() {
    std::vector<TYPE> t = {-1, -0.75, -0.5, 0, 0.25, 0.5, 0.75};
    std::vector<TYPE> y = {1, 0.8125, 0.75, 1, 1.3125, 1.75, 2.3125};
    Matrix<TYPE> A(7, 3);
    Matrix<TYPE> b(7, 1, y);
    for (int i = 0; i < 7; ++i) {
        A[i][0] = t[i] * t[i];
        A[i][1] = t[i];
        A[i][2] = 1;
    }
    QR_Solver<TYPE> solver;
    solver.compute(A);
    Matrix<TYPE> x = solver.solve(b);
    x.transpose().print();
    return 0;
}