//
// Created by Yanhong Liu on 2025/10/8.
//
#include "Condition_number.h"

#define TYPE long double

int main() {
    for (int N = 5; N <= 20; ++N) {
        Matrix<TYPE> A(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                A[i][j] = (TYPE) 1.0 / (i + j + 1);

        std::cout << "The condition number of " << N << "x" << N << " Hilbert matrix is " << condition_number_oo<TYPE>(A) << std::endl;

    }
    return 0;
}