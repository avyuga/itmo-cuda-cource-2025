
#ifndef UTILS
#define UTILS

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <Eigen/Dense>

using namespace std;


inline void init_random_generator() {
    srand(static_cast<unsigned>(time(NULL)));
}

template<typename T>
vector<vector<T>> generateMatrix(int matrix_size){
    vector<vector<T>> matrix(matrix_size, vector<T>(matrix_size));
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            matrix[i][j] = static_cast<T>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

template<typename T>
bool compareMatrices(const vector<vector<T>>& a, const vector<vector<T>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (a[i][j] != b[i][j]) return false;
        }
    }
    return true;
}

template<typename T>
Eigen::MatrixXd vectorToEigen(const vector<vector<T>>& data) {
    int rows = data.size();
    if (rows == 0) return Eigen::MatrixXd();

    int cols = data[0].size();
    Eigen::MatrixXd result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j];
        }
    }

    return result;
}

template<typename T>
void eigenToVector(const Eigen::MatrixXd &mat, vector<vector<T>> result) {
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            result[i][j] = mat(i, j);
        }
    }
}



template<typename T>
void print_matrix(const vector<vector<T>>& matrix, int precision = 4) {
    cout.precision(precision);
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            cout << fixed << value << "\t";
        }
        cout << "\n";
    }
}

#endif