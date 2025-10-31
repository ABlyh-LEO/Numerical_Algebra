//
// Created by Yanhong Liu on 2025/9/30.
//
#pragma once

#include "matrix_utils.h"

template<typename T>
class LU_Solver {
private:
    int _dim;

    Matrix<T> Res;

public:
    LU_Solver() = default;

    ~LU_Solver() = default;

    void compute(Matrix<T> &A) {
        if (A.cols() != A.rows()) {
            std::cerr << "Error(LU_Solver): Matrix is not square!" << std::endl;
            return;
        }
        Res = A;
        _dim = A.cols();
        for (int k = 0; k < _dim; ++k) {
            if (Res[k][k] == 0) {
                std::cerr << "Error(LU_Solver): Trying to use 0 as a divisor!" << std::endl;
                return;
            }
            for (int i = k + 1; i < _dim; ++i) {
                Res[i][k] /= Res[k][k];
                for (int j = k + 1; j < _dim; ++j) {
                    Res[i][j] -= Res[i][k] * Res[k][j];
                }
            }
        }
    }

    Matrix<T> solve(const Matrix<T> &b) {
        if (_dim <= 0) {
            std::cerr << "Error(LU_Solver): Matrix is not computed!" << std::endl;
            return Matrix<T>();
        }
        Matrix<T> y(_dim, 1), x(_dim, 1);
        for (int i = 0; i < _dim; ++i) {
            y[i][0] = b[i][0];
            for (int j = 0; j < i; ++j) {
                y[i][0] -= Res[i][j] * y[j][0];
            }
        }
        for (int i = _dim - 1; i >= 0; --i) {
            x[i][0] = y[i][0];
            for (int j = i + 1; j < _dim; ++j) {
                x[i][0] -= Res[i][j] * x[j][0];
            }
            x[i][0] /= Res[i][i];
        }
        return x;
    }
};

template<typename T>
class PLU_Solver {
private:
    int _dim;
    Matrix<T> Res;
    std::vector<int> P;
public:
    PLU_Solver() = default;

    ~PLU_Solver() = default;

    void compute(const Matrix<T> &A) {
        if (A.cols() != A.rows()) {
            std::cerr << "Error(PLU_Solver): Matrix is not square!" << std::endl;
            return;
        }
        _dim = A.cols();
        P.resize(_dim);
        Res = A;
        for (int i = 0; i < _dim; ++i) P[i] = i;
        for (int k = 0; k < _dim; ++k) {
            int maxIndex = k;
            for (int i = k + 1; i < _dim; ++i) {
                if (std::abs(Res[i][k]) > std::abs(Res[maxIndex][k])) {
                    maxIndex = i;
                }
            }
            if (Res[maxIndex][k] == 0) {
                std::cerr << "Error(PLU_Solver): Trying to use 0 as a divisor!" << std::endl;
                return;
            }
            if (maxIndex != k) {
                std::swap(P[k], P[maxIndex]);
                for (int j = 0; j < _dim; ++j) {
                    std::swap(Res[k][j], Res[maxIndex][j]);
                }
            }
            for (int i = k + 1; i < _dim; ++i) {
                Res[i][k] /= Res[k][k];
                for (int j = k + 1; j < _dim; ++j) {
                    Res[i][j] -= Res[i][k] * Res[k][j];
                }
            }
        }
    }

    Matrix<T> solve(const Matrix<T> &b) {
        if (_dim <= 0) {
            std::cerr << "Error(PLU_Solver): Matrix is not computed!" << std::endl;
            return Matrix<T>();
        }
        Matrix<T> y(_dim, 1), x(_dim, 1);
        for (int i = 0; i < _dim; ++i) {
            y[i][0] = b[P[i]][0];
            for (int j = 0; j < i; ++j) {
                y[i][0] -= Res[i][j] * y[j][0];
            }
        }
        for (int i = _dim - 1; i >= 0; --i) {
            x[i][0] = y[i][0];
            for (int j = i + 1; j < _dim; ++j) {
                x[i][0] -= Res[i][j] * x[j][0];
            }
            x[i][0] /= Res[i][i];
        }
        return x;
    }

    Matrix<T> solveT(const Matrix<T> &b) {// solve ATx = b
        if (_dim <= 0) {
            std::cerr << "Error(PLU_Solver): Matrix is not computed!" << std::endl;
            return Matrix<T>();
        }
        Matrix<T> z(_dim, 1), y(_dim, 1), x(_dim, 1);
        for (int i = 0; i < _dim; ++i) {
            z[i][0] = b[i][0];
            for (int j = 0; j < i; ++j) {
                z[i][0] -= Res[j][i] * z[j][0];
            }
            z[i][0] /= Res[i][i];
        }
        for (int i = _dim - 1; i >= 0; --i) {
            y[i][0] = z[i][0];
            for (int j = i + 1; j < _dim; ++j) {
                y[i][0] -= Res[j][i] * y[j][0];
            }
        }
        int invP[_dim];
        for (int i = 0; i < _dim; ++i) {
            invP[P[i]] = i;
        }

        for (int i = 0; i < _dim; ++i) {
            x[i][0] = y[invP[i]][0];
        }

        return x;
    }
};