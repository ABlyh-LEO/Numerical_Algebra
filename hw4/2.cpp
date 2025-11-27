//
// Created by Yanhong Liu on 2025/11/27.
//
#include "It_Method.h"
#include <iostream>
#include <chrono>

#define TYPE double
#define INF (1e9+7)

void prob2_sparse(int N) {
    TYPE h = 1.0 / N;
    auto idx = [&](int i, int j) {
        return i * (N + 1) + j;
    };

    auto f = [&](int i, int j) {
        return i * h + j * h;
    };

    auto g = [&](int i, int j) {
        return std::exp(i * h * j * h);
    };


    Matrix<TYPE> A((N + 1) * (N + 1), (N + 1) * (N + 1));
    Matrix<TYPE> b((N + 1) * (N + 1), 1);
    Matrix<TYPE> x((N + 1) * (N + 1), 1);
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            if (i == 0 || j == 0 || i == N || j == N) {
                A[idx(i, j)][idx(i, j)] = 1;
                b[idx(i, j)][0] = 1;
            } else {
                A[idx(i, j)][idx(i - 1, j)] = -1;
                A[idx(i, j)][idx(i, j - 1)] = -1;
                A[idx(i, j)][idx(i, j)] = 4 + h * h * g(i, j);
                A[idx(i, j)][idx(i + 1, j)] = -1;
                A[idx(i, j)][idx(i, j + 1)] = -1;
                b[idx(i, j)][0] = h * h * f(i, j);
            }
        }
    }
    SparseMatrix<TYPE> A_sparse(A);
    auto start = std::chrono::high_resolution_clock::now();
    auto its = GS_Method(A_sparse, b, x, INF, 1e-7);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//    for (int i = 0; i <= N; ++i) {
//        for (int j = 0; j <= N; ++j) {
//            std::cout << std::fixed << std::setprecision(2) << x[idx(i, j)][0] << " ";
//        }
//        std::cout << std::endl;
//    }
    //  b.print();
    std::cout << "---------------Sparse---------------" << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "iterations: " << its << std::endl;
    std::cout << "Function took " << duration.count() << " milliseconds to execute." << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

void prob2_dense(int N) {
    TYPE h = 1.0 / N;
    auto idx = [&](int i, int j) {
        return i * (N + 1) + j;
    };

    auto f = [&](int i, int j) {
        return i * h + j * h;
    };

    auto g = [&](int i, int j) {
        return std::exp(i * h * j * h);
    };


    Matrix<TYPE> A((N + 1) * (N + 1), (N + 1) * (N + 1));
    Matrix<TYPE> b((N + 1) * (N + 1), 1);
    Matrix<TYPE> x((N + 1) * (N + 1), 1);
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            if (i == 0 || j == 0 || i == N || j == N) {
                A[idx(i, j)][idx(i, j)] = 1;
                b[idx(i, j)][0] = 1;
            } else {
                A[idx(i, j)][idx(i - 1, j)] = -1;
                A[idx(i, j)][idx(i, j - 1)] = -1;
                A[idx(i, j)][idx(i, j)] = 4 + h * h * g(i, j);
                A[idx(i, j)][idx(i + 1, j)] = -1;
                A[idx(i, j)][idx(i, j + 1)] = -1;
                b[idx(i, j)][0] = h * h * f(i, j);
            }
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto its = GS_Method(A, b, x, INF, 1e-7);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//    for (int i = 0; i <= N; ++i) {
//        for (int j = 0; j <= N; ++j) {
//            std::cout << std::fixed << std::setprecision(2) << x[idx(i, j)][0] << " ";
//        }
//        std::cout << std::endl;
//    }
    //  b.print();
    std::cout << "---------------Dense----------------" << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "iterations: " << its << std::endl;
    std::cout << "Function took " << duration.count() << " milliseconds to execute." << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

int main() {
    prob2_sparse(20);
    prob2_sparse(40);
    prob2_sparse(80);
    prob2_dense(20);
    prob2_dense(40);
    prob2_dense(80);
    return 0;
}