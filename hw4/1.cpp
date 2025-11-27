//
// Created by Yanhong Liu on 2025/11/27.
//
#include "It_Method.h"
#include <iostream>

#define TYPE double
#define INF (1e9+7)

void prob1(TYPE _eps) {
    TYPE eps = _eps;
    int n = 100;
    TYPE h = 1.0 / n;
    TYPE a = 1.0 / 2;
    Matrix<TYPE> A(101, 101);
    for (int i = 0; i < 101; ++i) {
        for (int j = 0; j < 101; ++j) {
            if (i == j) {
                A[i][j] = -(2 * eps + h);
            } else if (i == j + 1) {
                A[i][j] = eps;
            } else if (i == j - 1) {
                A[i][j] = eps + h;
            }
        }
    }
    A[0][0] = 1, A[0][1] = 0;
    A[100][100] = 1, A[100][99] = 0;
    Matrix<TYPE> b(101, 1);
    for (int i = 0; i < 101; ++i) {
        b[i][0] = a * h * h;
    }
    b[0][0] = 0;
    b[100][0] = 1;
    Matrix<TYPE> x(101, 1);
    Matrix<TYPE> true_ans(101, 1);
    for (int i = 0; i <= 100; ++i) {
        TYPE xi = i * h;
        true_ans[i][0] = (1 - a) / (1 - std::exp(-1 / eps)) * (1 - std::exp(-xi / eps)) + a * xi;
    }
    //true_ans.transpose().print();

    //A.print();
    //b.transpose().print();
    std::cout << "------------------------------------" << std::endl;
    std::cout << "eps: " << eps << std::endl;
    Jacobi_Method(A, b, x, INF);
    std::cout << "Jacobi Method Error: " << (x - true_ans).norm_oo() / true_ans.norm_oo() << std::endl;
    x.set_shape(101, 1);
    GS_Method(A, b, x, INF);
    std::cout << "G-S Method Error: " << (x - true_ans).norm_oo() / true_ans.norm_oo() << std::endl;
    x.set_shape(101, 1);
    SOR_Method(A, b, x, INF);
    std::cout << "SOR Method Error: " << (x - true_ans).norm_oo() / true_ans.norm_oo() << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

int main() {
    prob1(1.0);
    prob1(0.1);
    prob1(0.01);
    prob1(0.0001);
    return 0;
}