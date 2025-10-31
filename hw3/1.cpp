//
// Created by Yanhong Liu on 2025/10/30.
//
#include <random>
#include "matrix_utils.h"
#include "LU_solver.h"
#include "QR_solver.h"
#include "Cholesky_solver.h"

#define TYPE long double
#define SHOW_RESULT 0

int main() {
    LU_Solver<TYPE> lu_solver;
    PLU_Solver<TYPE> plu_solver;
    QR_Solver<TYPE> qr_solver;
    Cholesky_solver<TYPE> cholesky_solver;
    Better_Cholesky_solver<TYPE> better_cholesky_solver;

    {
        auto A = Matrix<TYPE>(86, 86);
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
        auto b = Matrix<TYPE>(86, 1);
        b[0][0] = 7;
        for (int i = 1; i < 85; ++i) {
            b[i][0] = 15;
        }
        b[85][0] = 14;
        auto real = Matrix<TYPE>(86, 1);
        for (int i = 0; i < 86; ++i) {
            real[i][0] = 1;
        }
        lu_solver.compute(A);
        plu_solver.compute(A);
        qr_solver.compute(A);
        std::cout << "Real solution:" << std::endl;
        if (SHOW_RESULT)real.transpose().print();
        std::cout << std::endl;

        {
            auto x = lu_solver.solve(b);
            std::cout << "LU solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
        {
            auto x = plu_solver.solve(b);
            std::cout << "PLU solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
        {
            auto x = qr_solver.solve(b);
            std::cout << "QR solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
        std::cout << "-----------------------------------------------------------------------------" << std::endl;
    }
    {
        auto A = Matrix<TYPE>(100, 100);
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
        auto x = Matrix<TYPE>(100, 1);
        auto new_rand = []() -> TYPE {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<TYPE> dis(-50.0, 50.0);
            return dis(gen);
        };
        for (int i = 0; i < 100; ++i) {
            x[i][0] = new_rand();
        }
        auto b = A * x;
        lu_solver.compute(A);
        plu_solver.compute(A);
        qr_solver.compute(A);
        cholesky_solver.compute(A);
        better_cholesky_solver.compute(A);
        std::cout << "Real solution:" << std::endl;
        if (SHOW_RESULT)x.transpose().print();
        std::cout << std::endl;
        {
            auto x_ = lu_solver.solve(b);
            std::cout << "LU solution:" << std::endl;
            if (SHOW_RESULT)x_.transpose().print();
            std::cout << "Error: " << (x_ - x).norm_oo() / x.norm_oo() << std::endl << std::endl;
        }
        {
            auto x_ = plu_solver.solve(b);
            std::cout << "PLU solution:" << std::endl;
            if (SHOW_RESULT)x_.transpose().print();
            std::cout << "Error: " << (x_ - x).norm_oo() / x.norm_oo() << std::endl << std::endl;
        }
        {
            auto x_ = qr_solver.solve(b);
            std::cout << "QR solution:" << std::endl;
            if (SHOW_RESULT)x_.transpose().print();
            std::cout << "Error: " << (x_ - x).norm_oo() / x.norm_oo() << std::endl << std::endl;
        }
        {
            auto x_ = cholesky_solver.solve(b);
            std::cout << "Cholesky solution:" << std::endl;
            if (SHOW_RESULT)x_.transpose().print();
            std::cout << "Error: " << (x_ - x).norm_oo() / x.norm_oo() << std::endl << std::endl;
        }
        {
            auto x_ = better_cholesky_solver.solve(b);
            std::cout << "Cholesky(LDLT) solution:" << std::endl;
            if (SHOW_RESULT)x_.transpose().print();
            std::cout << "Error: " << (x_ - x).norm_oo() / x.norm_oo() << std::endl << std::endl;
        }
        std::cout << "-----------------------------------------------------------------------------" << std::endl;
    }
    {
        Matrix<TYPE> A(40, 40);
        for (int i = 0; i < 40; ++i) {
            for (int j = 0; j < 40; ++j) {
                A[i][j] = 1.0 / (i + j + 1);
            }
        }
        Matrix<TYPE> b(40, 1);
        for (int i = 0; i < 40; ++i) {
            for (int j = 0; j < 40; ++j) {
                b[i][0] += 1.0 / (i + j + 1);
            }
        }
        lu_solver.compute(A);
        plu_solver.compute(A);
        qr_solver.compute(A);
        cholesky_solver.compute(A);
        better_cholesky_solver.compute(A);
        auto real = Matrix<TYPE>(40, 1);
        for (int i = 0; i < 40; ++i) {
            real[i][0] = 1;
        }
        std::cout << "Real solution:" << std::endl;
        if (SHOW_RESULT)real.transpose().print();
        std::cout << std::endl;
        {
            auto x = lu_solver.solve(b);
            std::cout << "LU solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
        {
            auto x = plu_solver.solve(b);
            std::cout << "PLU solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
        {
            auto x = qr_solver.solve(b);
            std::cout << "QR solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
        {
            auto x = cholesky_solver.solve(b);
            std::cout << "Cholesky solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
        {
            auto x = better_cholesky_solver.solve(b);
            std::cout << "Cholesky(LDLT) solution:" << std::endl;
            if (SHOW_RESULT)x.transpose().print();
            std::cout << "Error: " << (x - real).norm_oo() / real.norm_oo() << std::endl << std::endl;
        }
    }

    return 0;
}