//
// Created by Yanhong Liu on 2025/10/8.
//
#include "Condition_number.h"
#include <random>

#define TYPE long double

template<int N>
void makeA() {
    Matrix<TYPE, N, N> A;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i < j && j != N - 1) A[i][j] = 0;
            else if (i == j || j == N - 1) A[i][j] = 1;
            else A[i][j] = -1;
        }
    Matrix<TYPE, N, 1> x;

    auto new_rand = []() -> TYPE {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<TYPE> dis(-50.0, 50.0);
        return dis(gen);
    };

    for (int i = 0; i < N; ++i) {
        x[i][0] = new_rand();
    }
    auto b = A * x;
    PLU_Solver<TYPE, N> solver;
    solver.compute(A);
    auto x_ = solver.solve(b);
    std::cout << "n = " << N << " , estimated error = " << condition_number_oo(A) * (b - A * x_).oo_norm() / b.oo_norm() << ", real error = " << (x - x_).oo_norm() / x.oo_norm() << std::endl;
}

template<int... Ns>
void runAll(std::integer_sequence<int, Ns...>) {
    (makeA<Ns>(), ...);
}

int main() {
    runAll(std::integer_sequence<int, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>{});
    return 0;
}