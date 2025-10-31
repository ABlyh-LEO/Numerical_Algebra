//
// Created by Yanhong Liu on 2025/10/8.
//
#include "Condition_number.h"
#include <random>

#define TYPE long double

int main() {
    for (int N = 5; N <= 30; ++N) {
        Matrix<TYPE> A(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (i < j && j != N - 1) A[i][j] = 0;
                else if (i == j || j == N - 1) A[i][j] = 1;
                else A[i][j] = -1;
            }
        Matrix<TYPE> x(N, 1);

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
        PLU_Solver<TYPE> solver;
        solver.compute(A);
        auto x_ = solver.solve(b);
        std::cout << "n = " << N << " , estimated error = " << condition_number_oo(A) * (b - A * x_).norm_oo() / b.norm_oo() << ", real error = " << (x - x_).norm_oo() / x.norm_oo() << std::endl;

    }
    return 0;
}