//
// Created by Yanhong Liu on 2025/10/8.
//
#include "Condition_number.h"

#define TYPE long double

template<int N>
void makeA() {
    Matrix<TYPE, N, N> A;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i][j] = (TYPE) 1.0 / (i + j + 1);

    std::cout << "The condition number of " << N << "x" << N << " Hilbert matrix is " << condition_number_oo<TYPE, N>(A) << std::endl;
}

template<int... Ns>
void runAll(std::integer_sequence<int, Ns...>) {
    (makeA<Ns>(), ...);
}

int main() {
    runAll(std::integer_sequence<int, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20>{});
    return 0;
}