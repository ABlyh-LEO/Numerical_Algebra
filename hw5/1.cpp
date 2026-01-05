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
    int n = 20;
    TYPE h = 1.0 / n;
    TYPE h2_4 = h * h / 4.0;
    TYPE coef_center = 1.0 + h2_4;
    TYPE coef_neighbor = -0.25;

    auto idx = [&](int i, int j) {
        return (j - 1) * (n - 1) + (i - 1);
    };

    auto f = [&](int i, int j) {
        TYPE x = i * h;
        TYPE y = j * h;
        return std::sin(x * y);
    };

    auto phi = [&](int i, int j) {
        TYPE x = i * h;
        TYPE y = j * h;
        return x * x + y * y;
    };

    int size = (n - 1) * (n - 1);
    Matrix<TYPE> A(size, size);
    Matrix<TYPE> b(size, 1);

    for (int j = 1; j <= n - 1; ++j) {
        for (int i = 1; i <= n - 1; ++i) {
            int row = idx(i, j);

            A[row][row] = coef_center;

            if (i > 1) {
                A[row][idx(i - 1, j)] = coef_neighbor;
            }

            if (i < n - 1) {
                A[row][idx(i + 1, j)] = coef_neighbor;
            }

            if (j > 1) {
                A[row][idx(i, j - 1)] = coef_neighbor;
            }

            if (j < n - 1) {
                A[row][idx(i, j + 1)] = coef_neighbor;
            }

            b[row][0] = h2_4 * f(i, j);

            if (i == 1) {
                b[row][0] += 0.25 * phi(0, j);
            }
            if (i == n - 1) {
                b[row][0] += 0.25 * phi(n, j);
            }
            if (j == 1) {
                b[row][0] += 0.25 * phi(i, 0);
            }
            if (j == n - 1) {
                b[row][0] += 0.25 * phi(i, n);
            }
        }
    }

    SparseMatrix<TYPE> A_sparse(A);

    Matrix<TYPE> x_cg(size, 1);

    auto start = std::chrono::high_resolution_clock::now();
    auto its_cg = CG_Method(A_sparse, b, x_cg, INF, 1e-4);
    auto end = std::chrono::high_resolution_clock::now();
    auto time_cg = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << its_cg << std::endl;
    std::cout << time_cg.count() << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < x_cg.rows(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << x_cg[i][0] << ",";
    }
    return 0;
}
