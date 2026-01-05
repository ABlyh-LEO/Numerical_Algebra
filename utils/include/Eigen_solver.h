//
// Created by Yanhong Liu on 2025/12/28.
//

#ifndef NUMERICAL_ALGEBRA_EIGEN_SOLVER_H
#define NUMERICAL_ALGEBRA_EIGEN_SOLVER_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <vector>

#include "matrix_utils.h"

// Build Companion Matrix
// For polynomial f(x) = x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0 = 0
// Companion matrix is:
//   [ 0  0  0  ...  0  -a_0   ]
//   [ 1  0  0  ...  0  -a_1   ]
//   [ 0  1  0  ...  0  -a_2   ]
//   [ ...                     ]
//   [ 0  0  0  ...  1  -a_{n-1}]
template <typename T>
Matrix<T> buildCompanionMatrix(const std::vector<T>& coeffs) {
  int n = coeffs.size();  // polynomial degree
  Matrix<T> C(n, n);
  for (int i = 0; i < n; ++i) {
    C[i][n - 1] = -coeffs[i];
  }
  for (int i = 1; i < n; ++i) {
    C[i][i - 1] = 1;
  }
  return C;
}

// Power method to find the largest eigenvalue by magnitude
// Returns: pair<eigenvalue, eigenvector>
template <typename T>
std::pair<T, Matrix<T>> powerMethod(const Matrix<T>& A, int maxIter = 10000,
                                    T tol = 1e-10) {
  int n = A.rows();
  Matrix<T> u(n, 1);
  // Initialize vector
  for (int i = 0; i < n; ++i) {
    u[i][0] = 1.0;
  }

  T lambda = 0;
  T lambdaOld = 0;

  for (int k = 0; k < maxIter; ++k) {
    Matrix<T> v = A * u;

    // Find component with maximum magnitude
    T maxVal = 0;
    int maxIdx = 0;
    for (int i = 0; i < n; ++i) {
      if (std::abs(v[i][0]) > std::abs(maxVal)) {
        maxVal = v[i][0];
        maxIdx = i;
      }
    }

    lambda = maxVal;

    // Normalize
    for (int i = 0; i < n; ++i) {
      u[i][0] = v[i][0] / lambda;
    }

    // Check convergence
    if (std::abs(lambda - lambdaOld) < tol * std::abs(lambda)) {
      return {lambda, u};
    }
    lambdaOld = lambda;
  }

  return {lambda, u};
}

// Find the largest polynomial root by magnitude using companion matrix + power
// method coeffs: [a_0, a_1, ..., a_{n-1}] for x^n + a_{n-1}*x^{n-1} + ... +
// a_1*x + a_0 = 0
template <typename T>
T findLargestRootByPowerMethod(const std::vector<T>& coeffs,
                               int maxIter = 10000, T tol = 1e-10) {
  Matrix<T> C = buildCompanionMatrix(coeffs);
  auto [lambda, _] = powerMethod(C, maxIter, tol);
  return lambda;
}

// Givens rotation
template <typename T>
void givensRotation(T a, T b, T& c, T& s) {
  if (b == 0) {
    c = 1;
    s = 0;
  } else if (std::abs(b) > std::abs(a)) {
    T tau = -a / b;
    s = 1 / std::sqrt(1 + tau * tau);
    c = s * tau;
  } else {
    T tau = -b / a;
    c = 1 / std::sqrt(1 + tau * tau);
    s = c * tau;
  }
}

// Householder transformation helper function
template <typename T>
std::pair<Matrix<T>, T> householderForEigen(const Matrix<T>& x) {
  int n = x.rows();
  Matrix<T> v(n, 1);

  T sigma = 0;
  for (int i = 1; i < n; ++i) {
    sigma += x[i][0] * x[i][0];
  }

  for (int i = 0; i < n; ++i) {
    v[i][0] = x[i][0];
  }

  T beta;
  if (sigma == 0 && x[0][0] >= 0) {
    beta = 0;
  } else if (sigma == 0 && x[0][0] < 0) {
    beta = 2;
    v[0][0] = 1;
  } else {
    T mu = std::sqrt(x[0][0] * x[0][0] + sigma);
    if (x[0][0] <= 0) {
      v[0][0] = x[0][0] - mu;
    } else {
      v[0][0] = -sigma / (x[0][0] + mu);
    }
    beta = 2 * v[0][0] * v[0][0] / (sigma + v[0][0] * v[0][0]);
    T denom = v[0][0];
    for (int i = 0; i < n; ++i) {
      v[i][0] /= denom;
    }
  }

  return {v, beta};
}

// Reduce matrix to upper Hessenberg form
template <typename T>
Matrix<T> toHessenberg(const Matrix<T>& A) {
  int n = A.rows();
  Matrix<T> H = A;

  for (int k = 0; k < n - 2; ++k) {
    // Construct Householder transformation to eliminate H[k+2:n, k]
    Matrix<T> x(n - k - 1, 1);
    for (int i = k + 1; i < n; ++i) {
      x[i - k - 1][0] = H[i][k];
    }

    auto [v, beta] = householderForEigen(x);

    // Left multiply: H[k+1:n, k:n] = (I - beta*v*v^T) * H[k+1:n, k:n]
    for (int j = k; j < n; ++j) {
      T dot = 0;
      for (int i = k + 1; i < n; ++i) {
        dot += v[i - k - 1][0] * H[i][j];
      }
      for (int i = k + 1; i < n; ++i) {
        H[i][j] -= beta * v[i - k - 1][0] * dot;
      }
    }

    // Right multiply: H[0:n, k+1:n] = H[0:n, k+1:n] * (I - beta*v*v^T)
    for (int i = 0; i < n; ++i) {
      T dot = 0;
      for (int j = k + 1; j < n; ++j) {
        dot += H[i][j] * v[j - k - 1][0];
      }
      for (int j = k + 1; j < n; ++j) {
        H[i][j] -= beta * dot * v[j - k - 1][0];
      }
    }
  }

  return H;
}

// Single step of implicit QR iteration with shift
template <typename T>
void implicitQRStep(Matrix<T>& H, int p, int q, T shift) {
  int n = q - p + 1;
  if (n <= 1) return;

  // First Givens rotation
  T c, s;
  T a = H[p][p] - shift;
  T b = H[p + 1][p];
  givensRotation(a, b, c, s);

  // Apply to left side of H
  for (int j = p; j < q + 1; ++j) {
    T t1 = H[p][j];
    T t2 = H[p + 1][j];
    H[p][j] = c * t1 - s * t2;
    H[p + 1][j] = s * t1 + c * t2;
  }

  // Apply to right side of H
  int imax = std::min(p + 2, q);
  for (int i = p; i <= imax; ++i) {
    T t1 = H[i][p];
    T t2 = H[i][p + 1];
    H[i][p] = c * t1 - s * t2;
    H[i][p + 1] = s * t1 + c * t2;
  }

  // Chase the bulge
  for (int k = p + 1; k < q; ++k) {
    givensRotation(H[k][k - 1], H[k + 1][k - 1], c, s);

    // Left multiply
    for (int j = k - 1; j < q + 1; ++j) {
      T t1 = H[k][j];
      T t2 = H[k + 1][j];
      H[k][j] = c * t1 - s * t2;
      H[k + 1][j] = s * t1 + c * t2;
    }

    // Right multiply
    int imin = std::min(k + 2, q);
    for (int i = p; i <= imin; ++i) {
      T t1 = H[i][k];
      T t2 = H[i][k + 1];
      H[i][k] = c * t1 - s * t2;
      H[i][k + 1] = s * t1 + c * t2;
    }
  }
}

// Francis double-shift QR algorithm
template <typename T>
void francisQRStep(Matrix<T>& H, int p, int q) {
  int n = q - p + 1;
  if (n <= 2) return;

  // Compute shifts
  T a = H[q - 1][q - 1];
  T b = H[q - 1][q];
  T c = H[q][q - 1];
  T d = H[q][q];
  T tr = a + d;
  T det = a * d - b * c;

  // First column
  T x = H[p][p] * H[p][p] + H[p][p + 1] * H[p + 1][p] - tr * H[p][p] + det;
  T y = H[p + 1][p] * (H[p][p] + H[p + 1][p + 1] - tr);
  T z = H[p + 1][p] * H[p + 2][p + 1];

  for (int k = p; k <= q - 2; ++k) {
    // Householder transformation
    Matrix<T> v(3, 1);
    v[0][0] = x;
    v[1][0] = y;
    v[2][0] = z;

    auto [u, beta] = householderForEigen(v);

    int r = std::max(0, k - 1);

    // Left multiply
    for (int j = r; j <= q; ++j) {
      T dot = 0;
      for (int i = 0; i < 3 && k + i <= q; ++i) {
        dot += u[i][0] * H[k + i][j];
      }
      for (int i = 0; i < 3 && k + i <= q; ++i) {
        H[k + i][j] -= beta * u[i][0] * dot;
      }
    }

    // Right multiply
    int s = std::min(k + 3, q);
    for (int i = p; i <= s; ++i) {
      T dot = 0;
      for (int j = 0; j < 3 && k + j <= q; ++j) {
        dot += H[i][k + j] * u[j][0];
      }
      for (int j = 0; j < 3 && k + j <= q; ++j) {
        H[i][k + j] -= beta * dot * u[j][0];
      }
    }

    x = H[k + 1][k];
    y = H[k + 2][k];
    if (k < q - 2) {
      z = H[k + 3][k];
    }
  }

  // Last 2x2 block
  T cc, ss;
  givensRotation(x, y, cc, ss);

  for (int j = q - 2; j <= q; ++j) {
    T t1 = H[q - 1][j];
    T t2 = H[q][j];
    H[q - 1][j] = cc * t1 - ss * t2;
    H[q][j] = ss * t1 + cc * t2;
  }

  for (int i = p; i <= q; ++i) {
    T t1 = H[i][q - 1];
    T t2 = H[i][q];
    H[i][q - 1] = cc * t1 - ss * t2;
    H[i][q] = ss * t1 + cc * t2;
  }
}

// Implicit QR algorithm to find all eigenvalues
// Returns vector of complex eigenvalues
template <typename T>
std::vector<std::complex<T>> implicitQREigenvalues(const Matrix<T>& A,
                                                   int maxIter = 1000,
                                                   T tol = 1e-10) {
  int n = A.rows();
  Matrix<T> H = toHessenberg(A);
  std::vector<std::complex<T>> eigenvalues;

  int q = n - 1;
  int iter = 0;

  while (q > 0 && iter < maxIter * n) {
    // Check convergence: whether H[q][q-1] is small enough
    T off = std::abs(H[q][q - 1]);
    T diag = std::abs(H[q - 1][q - 1]) + std::abs(H[q][q]);

    if (off < tol * diag || off < tol) {
      // 1x1 block converged
      eigenvalues.push_back(std::complex<T>(H[q][q], 0));
      H[q][q - 1] = 0;
      q--;
      iter = 0;
    } else if (q > 1) {
      T off2 = std::abs(H[q - 1][q - 2]);
      T diag2 = std::abs(H[q - 2][q - 2]) + std::abs(H[q - 1][q - 1]);

      if (off2 < tol * diag2 || off2 < tol) {
        // Check 2x2 block
        T a = H[q - 1][q - 1];
        T b = H[q - 1][q];
        T c = H[q][q - 1];
        T d = H[q][q];

        T tr = a + d;
        T det = a * d - b * c;
        T disc = tr * tr - 4 * det;

        if (disc >= 0) {
          T sqrtDisc = std::sqrt(disc);
          eigenvalues.push_back(std::complex<T>((tr + sqrtDisc) / 2, 0));
          eigenvalues.push_back(std::complex<T>((tr - sqrtDisc) / 2, 0));
        } else {
          T sqrtDisc = std::sqrt(-disc);
          eigenvalues.push_back(std::complex<T>(tr / 2, sqrtDisc / 2));
          eigenvalues.push_back(std::complex<T>(tr / 2, -sqrtDisc / 2));
        }

        if (q > 1) {
          H[q - 1][q - 2] = 0;
        }
        q -= 2;
        iter = 0;
      } else {
        // Execute one QR iteration step
        // Wilkinson shift
        T a = H[q - 1][q - 1];
        T b = H[q - 1][q];
        T c = H[q][q - 1];
        T d = H[q][q];
        T tr = a + d;
        T det = a * d - b * c;
        T disc = tr * tr - 4 * det;

        T shift;
        if (disc >= 0) {
          T sqrtDisc = std::sqrt(disc);
          T e1 = (tr + sqrtDisc) / 2;
          T e2 = (tr - sqrtDisc) / 2;
          shift = (std::abs(e1 - d) < std::abs(e2 - d)) ? e1 : e2;
        } else {
          shift = d;
        }

        // Find active block
        int p = 0;
        for (int i = q - 1; i >= 1; --i) {
          T offI = std::abs(H[i][i - 1]);
          T diagI = std::abs(H[i - 1][i - 1]) + std::abs(H[i][i]);
          if (offI < tol * diagI || offI < tol) {
            p = i;
            break;
          }
        }

        implicitQRStep(H, p, q, shift);
        iter++;
      }
    } else {
      // q == 1, handle 2x2 block
      T a = H[0][0];
      T b = H[0][1];
      T c = H[1][0];
      T d = H[1][1];

      T tr = a + d;
      T det = a * d - b * c;
      T disc = tr * tr - 4 * det;

      if (disc >= 0) {
        T sqrtDisc = std::sqrt(disc);
        eigenvalues.push_back(std::complex<T>((tr + sqrtDisc) / 2, 0));
        eigenvalues.push_back(std::complex<T>((tr - sqrtDisc) / 2, 0));
      } else {
        T sqrtDisc = std::sqrt(-disc);
        eigenvalues.push_back(std::complex<T>(tr / 2, sqrtDisc / 2));
        eigenvalues.push_back(std::complex<T>(tr / 2, -sqrtDisc / 2));
      }
      q = -1;
    }
  }

  if (q == 0) {
    eigenvalues.push_back(std::complex<T>(H[0][0], 0));
  }

  return eigenvalues;
}

// Build companion matrix from polynomial coefficients and find all roots using
// QR coeffs: [a_0, a_1, ..., a_{n-1}] for x^n + a_{n-1}*x^{n-1} + ... + a_1*x +
// a_0 = 0
template <typename T>
std::vector<std::complex<T>> findAllPolynomialRoots(
    const std::vector<T>& coeffs, int maxIter = 1000, T tol = 1e-10) {
  Matrix<T> C = buildCompanionMatrix(coeffs);
  return implicitQREigenvalues(C, maxIter, tol);
}

// Helper function: sort eigenvalues by magnitude
template <typename T>
void sortEigenvaluesByMagnitude(std::vector<std::complex<T>>& eigenvalues,
                                bool descending = true) {
  if (descending) {
    std::sort(eigenvalues.begin(), eigenvalues.end(),
              [](const std::complex<T>& a, const std::complex<T>& b) {
                return std::abs(a) > std::abs(b);
              });
  } else {
    std::sort(eigenvalues.begin(), eigenvalues.end(),
              [](const std::complex<T>& a, const std::complex<T>& b) {
                return std::abs(a) < std::abs(b);
              });
  }
}

// Print complex number
template <typename T>
void printComplex(const std::complex<T>& c, int precision = 10) {
  std::cout << std::fixed << std::setprecision(precision);
  if (std::abs(c.imag()) < 1e-12) {
    std::cout << c.real();
  } else if (c.imag() >= 0) {
    std::cout << c.real() << " + " << c.imag() << "i";
  } else {
    std::cout << c.real() << " - " << std::abs(c.imag()) << "i";
  }
}

#endif  // NUMERICAL_ALGEBRA_EIGEN_SOLVER_H
