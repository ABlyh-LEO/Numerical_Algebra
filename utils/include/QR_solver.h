// Created by Yanhong Liu on 2025/10/21.
// 修正版：修复 Householder 符号选择、v/beta 约定与数值稳定性
#ifndef NUMERICAL_ALGEBRA_QR_SOLVER_H
#define NUMERICAL_ALGEBRA_QR_SOLVER_H

#include "matrix_utils.h"
#include <cmath>
#include <limits>
#include <iostream>

#ifndef DEBUG_QR
#define DEBUG_QR 1
#endif

template<typename T>
std::pair<Matrix<T>, T> householder_reflection(const Matrix<T> &x) {
    int _n = x.rows();
    Matrix<T> tmp = x;

    // eta 用于缩放以防止 overflow/underflow
    T eta = tmp.norm_oo();
    if (eta == 0) {
        // 全零向量：返回 v 为零向量、beta=0 (等价于 H = I)
        return {Matrix<T>(_n, 1), static_cast<T>(0)};
    }
    tmp /= eta;

    // sigma = sum_{i=1..n-1} tmp[i]^2
    T sigma = static_cast<T>(0);
    for (int i = 1; i < _n; ++i) {
        T val = tmp[i][0];
        sigma += val * val;
    }

    Matrix<T> v(_n, 1);
    for (int i = 1; i < _n; ++i) {
        v[i][0] = tmp[i][0];
    }

    T beta = static_cast<T>(0);

    if (sigma == 0) {
        // x 已经是 e1 的倍数：不需要反射
        // 统一约定 v[0]=1, beta=0 (H=I)
        v[0][0] = static_cast<T>(1);
        beta = static_cast<T>(0);
        if (DEBUG_QR) {
            std::cout << "[HH] sigma==0 -> v[0]=1, beta=0\n";
        }
    } else {
        T x0 = tmp[0][0];
        T norm = std::sqrt(x0 * x0 + sigma);
        // 选择带符号的 alpha，避免取消误差：alpha = -sign(x0) * norm
        T alpha = (x0 >= static_cast<T>(0)) ? -norm : norm;
        T v0 = x0 - alpha;

        const T eps = std::numeric_limits<T>::epsilon() * static_cast<T>(100);
        if (std::abs(v0) < eps) {
            // 如果 v0 非常小，则退化为 identity（数值不稳定）
            v[0][0] = static_cast<T>(1);
            beta = static_cast<T>(0);
            if (DEBUG_QR) {
                std::cout << "[HH] v0 small -> treat as beta=0\n";
            }
        } else {
            // 先设置 v[0] = v0，然后将全部 v 除以 v0 使得 v[0]=1（方便存储）
            v[0][0] = v0;
            for (int i = 0; i < _n; ++i) {
                v[i][0] /= v0;
            }
            // beta = 2 * v0^2 / (sigma + v0^2)
            beta = static_cast<T>(2) * (v0 * v0) / (sigma + v0 * v0);
            // 另一种等价写法：beta = 2 / (v^T v)  （此处 v 已被缩放，形式一致）
            if (DEBUG_QR) {
                std::cout << "[HH] computed beta = " << beta << "\n";
            }
        }
    }

    return {v, beta};
}

template<typename T>
class QR_Solver {
private:
    int _n = 0;
    int _m = 0;
    Matrix<T> RES;   // 存放变换后的 A：上三角为 R，下三角存储 v 的下半部分（v[1..]）
    Matrix<T> d;     // 存放每步的 beta（长度 m）
public:
    QR_Solver() = default;

    ~QR_Solver() = default;

    void compute(const Matrix<T> &A) {
        _n = A.rows();
        _m = A.cols();
        d = Matrix<T>(_m, 1);
        RES = A;

        for (int j = 0; j < _m; ++j) {
            if (j >= _n) break;

            // 构造 x = RES[j..n-1, j]
            Matrix<T> x(_n - j, 1);
            for (int i = j; i < _n; ++i) {
                x[i - j][0] = RES[i][j];
            }

            auto [v, beta] = householder_reflection<T>(x);
            d[j][0] = beta;

            if (beta != static_cast<T>(0)) {
                // 直接做 rank-1 更新：tmp = tmp - beta * v * (v^T * tmp)
                Matrix<T> tmp(_n - j, _m - j);
                for (int ii = j; ii < _n; ++ii) {
                    for (int k = j; k < _m; ++k) {
                        tmp[ii - j][k - j] = RES[ii][k];
                    }
                }
                // w = v^T * tmp  (1 x (m-j))
                Matrix<T> w = v.transpose() * tmp; // size 1 x (m-j)
                // tmp -= beta * v * w
                Matrix<T> update = v * w;
                update = update * beta;
                tmp = tmp - update;

                for (int ii = j; ii < _n; ++ii) {
                    for (int k = j; k < _m; ++k) {
                        RES[ii][k] = tmp[ii - j][k - j];
                    }
                }
            } else {
                // beta==0: H = I, 不改变子矩阵
                if (DEBUG_QR) {
                    std::cout << "[QR] step " << j << " beta==0, skip update\n";
                }
            }

            // 存储 v 的下半部分到 RES 下三角（v[0] 约定为 1）
            for (int ii = j + 1; ii < _n; ++ii) {
                RES[ii][j] = v[ii - j][0];
            }

            if (DEBUG_QR) {
                // 打印 R 的对角（当前 j）
                std::cout << "[QR] after step " << j << ", R(" << j << "," << j << ") = " << RES[j][j] << "\n";
            }
        }

#if DEBUG_QR
        // 最终输出前 min(n,m) 个对角
        int diag = std::min(_n, _m);
        std::cout << "[QR] final R diagonals:\n";
        for (int i = 0; i < diag; ++i) {
            std::cout << "R(" << i << "," << i << ")=" << RES[i][i] << "\n";
        }
#endif
    }

    Matrix<T> solve(const Matrix<T> &b) {
        if (_n <= 0 || _m <= 0) {
            std::cerr << "Error(QR_Solver): Matrix is not computed!" << std::endl;
            return Matrix<T>();
        }

        if (b.rows() != _n || b.cols() != 1) {
            std::cerr << "Error(QR_Solver): b must be of size n x 1\n";
            return Matrix<T>();
        }

        // Apply Q^T to b: y = Q^T b, using stored v and beta
        Matrix<T> y = b;
        for (int j = 0; j < _m; ++j) {
            if (j >= _n) break;
            Matrix<T> v(_n - j, 1);
            v[0][0] = static_cast<T>(1);
            for (int ii = j + 1; ii < _n; ++ii) {
                v[ii - j][0] = RES[ii][j];
            }
            T beta = d[j][0];
            if (beta == static_cast<T>(0)) continue;

            Matrix<T> tmp(_n - j, 1);
            for (int ii = j; ii < _n; ++ii) {
                tmp[ii - j][0] = y[ii][0];
            }
            T coeff = (v.transpose() * tmp)[0][0];
            coeff *= beta;
            tmp = tmp - v * coeff;
            for (int ii = j; ii < _n; ++ii) {
                y[ii][0] = tmp[ii - j][0];
            }
        }

        // 回代解上三角 R x = y[0..m-1]
        Matrix<T> x(_m, 1);
        for (int ii = _m - 1; ii >= 0; --ii) {
            if (ii >= _n) {
                // 当 m > n 时，R 的对角不存在：无法直接回代
                std::cerr << "Error(QR_Solver): system underdetermined or R not available (m > n). ii=" << ii << std::endl;
                return Matrix<T>();
            }
            T sum = y[ii][0];
            for (int j = ii + 1; j < _m; ++j) {
                sum -= RES[ii][j] * x[j][0];
            }
            T rii = RES[ii][ii];
            if (std::abs(rii) < std::numeric_limits<T>::epsilon() * static_cast<T>(10)) {
                std::cerr << "Error(QR_Solver): zero diagonal in R at index " << ii << " (value=" << rii << ")\n";
                return Matrix<T>();
            }
            x[ii][0] = sum / rii;
            if (ii == 0) break; // 避免 signed int --ii 后的无限循环
        }
        return x;
    }
};

#endif //NUMERICAL_ALGEBRA_QR_SOLVER_H
