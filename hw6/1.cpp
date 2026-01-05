#include <iomanip>
#include <iostream>
#include <vector>

#include "Eigen_solver.h"

#define TYPE double

void printPolynomial(const std::vector<TYPE> &coeffs) {
    int n = coeffs.size();
    std::cout << "x^" << n;
    for (int i = n - 1; i >= 0; --i) {
        if (coeffs[i] >= 0) {
            std::cout << " + " << coeffs[i];
        } else {
            std::cout << " - " << std::abs(coeffs[i]);
        }
        if (i > 0) {
            std::cout << "*x^" << i;
        }
    }
    std::cout << " = 0" << std::endl;
}

void solve(const std::vector<TYPE> &coeffs, const std::string &name) {
    std::cout << "------------------------------------" << std::endl;
    std::cout << name << ": ";
    printPolynomial(coeffs);

    // Build companion matrix
    Matrix<TYPE> C = buildCompanionMatrix(coeffs);
    //C.print();

    // Use power method to find the largest eigenvalue by magnitude
    auto [lambda, eigenvec] = powerMethod(C, 10000, 1e-12);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Largest root by power method: " << lambda << std::endl;

    // Verification using Horner's method (more numerically stable)
    // f(x) = x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
    // Horner: f(x) = ((...((1*x + a_{n-1})*x + a_{n-2})*x + ...)*x + a_1)*x + a_0
    TYPE x = lambda;
    TYPE val = 1.0;  // coefficient of x^n
    for (int i = coeffs.size() - 1; i >= 0; --i) {
        val = val * x + coeffs[i];
    }
    std::cout << "Verification f(lambda) = " << val << std::endl;
    std::cout << std::endl;
}

int main() {
    // (i) x^3 + x^2 - 5x + 3 = 0
    // coeffs [a_0, a_1, a_2] = [3, -5, 1]
    std::vector<TYPE> coeffs1 = {3.0, -5.0, 1.0};
    solve(coeffs1, "(i)");

    // (ii) x^3 - 3x - 1 = 0
    // coeffs [a_0, a_1, a_2] = [-1, -3, 0]
    std::vector<TYPE> coeffs2 = {-1.0, -3.0, 0.0};
    solve(coeffs2, "(ii)");

    // (iii) x^8 + 101x^7 + 208.01x^6 + 10891.01x^5 + 9802.08x^4
    //       + 79108.9x^3 - 99902x^2 + 790x - 1000 = 0
    // coeffs [a_0, a_1, ..., a_7]
    std::vector<TYPE> coeffs3 = {
            -1000.0,   // a_0 (constant term)
            790.0,     // a_1
            -99902.0,  // a_2
            79108.9,   // a_3
            9802.08,   // a_4
            10891.01,  // a_5
            208.01,    // a_6
            101.0      // a_7
    };
    solve(coeffs3, "(iii)");

    return 0;
}
