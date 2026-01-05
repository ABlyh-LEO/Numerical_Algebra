//
// Created by Yanhong Liu on 2025/12/21.
//
#include "It_Method.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>

#define TYPE double
#define INF (1e9+7)

int main() {
    int n = 40;
    Matrix<TYPE> A(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = 1.0 / (i + j + 1);
        }
    }
    Matrix<TYPE> b(n, 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b[i][0] += 1.0 / (i + j + 1) / 3.0;
        }
    }
    Matrix<TYPE> x_cg(n, 1);

    auto start = std::chrono::high_resolution_clock::now();
    auto its_cg = CG_Method(A, b, x_cg, INF, 1e-7);
    auto end = std::chrono::high_resolution_clock::now();
    auto time_cg = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << its_cg << std::endl;
    std::cout << time_cg.count() << std::endl;

    x_cg.transpose().print();

    Matrix<TYPE> x_true(n, 1);
    for (int i = 0; i < n; ++i) {
        x_true[i][0] = 1.0 / 3.0;
    }
    std::cout << "CG Method Error: " << (x_cg - x_true).norm_oo() / x_true.norm_oo() << std::endl;
    return 0;
}
