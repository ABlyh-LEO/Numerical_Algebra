// Created by Yanhong Liu on 2025/10/21.
//

#ifndef NUMERICAL_ALGEBRA_QR_SOLVER_H
#define NUMERICAL_ALGEBRA_QR_SOLVER_H

#include "matrix_utils.h"

template<typename T>
std::pair<Matrix<T>, T> householder(const Matrix<T> &x) {
    int _n = x.rows();
    auto tmp = x;
    T eta = tmp.norm_oo();
    if (eta == 0) {
        return {Matrix<T>(_n, 1), 0};
    }
    tmp /= eta;
    T sigma = 0;
    for (int i = 1; i < _n; ++i) {
        sigma += tmp[i][0] * tmp[i][0];
    }
    Matrix<T> v(_n, 1);
    for (int i = 1; i < _n; ++i) {
        v[i][0] = tmp[i][0];
    }
    T beta;
    if (sigma == 0) {
        beta = 0;
        v[0][0] = 1;
    } else {
        T alpha = std::sqrt(tmp[0][0] * tmp[0][0] + sigma);
        if (tmp[0][0] <= 0) {
            v[0][0] = tmp[0][0] - alpha;
        } else {
            v[0][0] = -sigma / (tmp[0][0] + alpha);
        }
        beta = 2 * v[0][0] * v[0][0] / (sigma + v[0][0] * v[0][0]);
        T divisor = v[0][0];
        v /= divisor;
    }
    return {v, beta};
}

template<typename T>
class QR_Solver {
private:
    int _n{};
    int _m{};
    Matrix<T> RES;
    Matrix<T> d;
public:
    QR_Solver() = default;

    ~QR_Solver() = default;

    void compute(const Matrix<T> &A) {
        _n = A.rows();
        _m = A.cols();
        RES = A;
        d.set_shape(_m, 1);
        for (int j = 0; j < _m; ++j) {
            if (j < _n - 1) {
                Matrix<T> x(_n - j, 1);
                for (int i = j; i < _n; ++i) {
                    x[i - j][0] = RES[i][j];
                }
                auto [v, beta] = householder(x);
                d[j][0] = beta;
                Matrix<T> tmp(_n - j, _m - j);
                for (int p = j; p < _n; ++p) {
                    for (int q = j; q < _m; ++q) {
                        tmp[p - j][q - j] = RES[p][q];
                    }
                }
                Matrix<T> w = v.transpose() * tmp;
                tmp = tmp - (v * w) * beta;
                for (int p = j; p < _n; ++p) {
                    for (int q = j; q < _m; ++q) {
                        RES[p][q] = tmp[p - j][q - j];
                    }
                }
                for (int i = j + 1; i < _n; ++i) {
                    RES[i][j] = v[i - j][0];
                }
            }
        }
    }

    Matrix<T> solve(const Matrix<T> &b) {
        Matrix<T> y = b;
        for (int j = 0; j < _m; ++j) {
            if (j < _n - 1) {
                Matrix<T> v(_n - j, 1);
                v[0][0] = 1;
                for (int i = j + 1; i < _n; ++i) {
                    v[i - j][0] = RES[i][j];
                }
                T beta = d[j][0];
                Matrix<T> tmp(_n - j, 1);
                for (int i = j; i < _n; ++i) {
                    tmp[i - j][0] = y[i][0];
                }
                T w_scalar = (v.transpose() * tmp)[0][0];
                tmp = tmp - (v * w_scalar) * beta;
                for (int i = j; i < _n; ++i) {
                    y[i][0] = tmp[i - j][0];
                }
            }
        }
        Matrix<T> x(_m, 1);
        for (int i = _m - 1; i >= 0; --i) {
            T sum = 0;
            for (int j = i + 1; j < _m; ++j) {
                sum += RES[i][j] * x[j][0];
            }
            x[i][0] = (y[i][0] - sum) / RES[i][i];
        }
        return x;
    }
};


#endif //NUMERICAL_ALGEBRA_QR_SOLVER_H