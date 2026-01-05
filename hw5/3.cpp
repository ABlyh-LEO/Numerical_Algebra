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
    TYPE tmp1[] = {10, 1, 2, 3, 4,
                   1, 9, -1, 2, -3,
                   2, -1, 7, 3, -5,
                   3, 2, 3, 12, -1,
                   4, -3, -5, -1, 15};
    Matrix<TYPE> A(5, 5, tmp1);
    TYPE tmp2[] = {12, -27, 14, -17, 12};
    Matrix<TYPE> b(5, 1, tmp2);


    Matrix<TYPE> x_j(5, 1);
    Matrix<TYPE> x_gs(5, 1);
    Matrix<TYPE> x_cg(5, 1);

    {
        auto start = std::chrono::high_resolution_clock::now();
        auto its_cg = CG_Method(A, b, x_cg, INF, 1e-7);
        auto end = std::chrono::high_resolution_clock::now();
        auto time_cg = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "--------------------CG------------------------" << std::endl;
        std::cout << its_cg << std::endl;
        std::cout << time_cg.count() << std::endl;
        x_cg.transpose().print();
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        auto its_gs = GS_Method(A, b, x_gs, INF, 1e-7);
        auto end = std::chrono::high_resolution_clock::now();
        auto time_gs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "--------------------GS------------------------" << std::endl;
        std::cout << its_gs << std::endl;
        std::cout << time_gs.count() << std::endl;
        x_gs.transpose().print();
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        auto its_j = Jacobi_Method(A, b, x_j, INF, 1e-7);
        auto end = std::chrono::high_resolution_clock::now();
        auto time_j = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "--------------------JACOBI--------------------" << std::endl;
        std::cout << its_j << std::endl;
        std::cout << time_j.count() << std::endl;
        x_j.transpose().print();
    }

    return 0;
}
