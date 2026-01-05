#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Eigen_solver.h"

#define TYPE double

// Find all roots of x^41 + x^3 + 1 = 0
void problem2() {
    std::cout << "============================================" << std::endl;
    std::cout << "(2) Find all roots of x^41 + x^3 + 1 = 0" << std::endl;
    std::cout << "============================================" << std::endl;

    // Build coefficient vector: x^41 + x^3 + 1 = 0
    // coeffs [a_0, a_1, ..., a_40]
    // a_0 = 1 (constant term), a_3 = 1, others are 0
    std::vector<TYPE> coeffs(41, 0.0);
    coeffs[0] = 1.0;  // constant term
    coeffs[3] = 1.0;  // x^3 term

    // Use implicit QR algorithm to find all roots
    auto roots = findAllPolynomialRoots(coeffs, 2000, 1e-12);

    // Sort by magnitude
    sortEigenvaluesByMagnitude(roots, true);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Total " << roots.size() << " roots:" << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < (int) roots.size(); ++i) {
        std::cout << "root[" << std::setw(2) << i + 1 << "] = ";
        printComplex(roots[i], 10);
        std::cout << "  (|root| = " << std::abs(roots[i]) << ")" << std::endl;
    }
    std::cout << std::endl;
}

// Find all eigenvalues of matrix A for x=0.9, 1.0, 1.1
void problem3() {
    std::cout << "============================================" << std::endl;
    std::cout << "(3) Find all eigenvalues of matrix A for x=0.9, 1.0, 1.1"
              << std::endl;
    std::cout << "============================================" << std::endl;

    std::vector<TYPE> xValues = {0.9, 1.0, 1.1};

    for (TYPE x: xValues) {
        std::cout << std::endl;
        std::cout << "------------------------------------" << std::endl;
        std::cout << "x = " << x << std::endl;

        // Build matrix A
        // A = [ 9.1  3.0  2.6  4.0 ]
        //     [ 4.2  5.3  4.7  1.6 ]
        //     [ 3.2  1.7  9.4   x  ]
        //     [ 6.1  4.9  3.5  6.2 ]
        Matrix<TYPE> A(4, 4);
        TYPE data[] = {9.1, 3.0, 2.6, 4.0, 4.2, 5.3, 4.7, 1.6,
                       3.2, 1.7, 9.4, x, 6.1, 4.9, 3.5, 6.2};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                A[i][j] = data[i * 4 + j];
            }
        }

        std::cout << "Matrix A:" << std::endl;
        A.print();

        // Use implicit QR to find eigenvalues
        auto eigenvalues = implicitQREigenvalues(A, 1000, 1e-12);

        // Sort by magnitude
        sortEigenvaluesByMagnitude(eigenvalues, true);

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Eigenvalues:" << std::endl;
        for (int i = 0; i < (int) eigenvalues.size(); ++i) {
            std::cout << "  lambda[" << i + 1 << "] = ";
            printComplex(eigenvalues[i], 10);
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
}

int main() {
    problem2();
    problem3();

    return 0;
}
