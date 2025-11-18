#include <iostream>
#include <vector>
#include <chrono>

#include "multiplication.h"

using namespace std;

// matrix: (rows, cols)
// A: (M, N), B: (N, P), C: (M, P)
int main(int argc, char* argv[]) {

    const int M = stoi(argv[1]);
    const int N = stoi(argv[2]);
    const int P = stoi(argv[3]);

    init_random_generator();

    auto A = generateMatrix<double>(M, N);
    auto B = generateMatrix<double>(N, P);
    
    cout << "CPU" << endl;
    vector<vector<double>> res_simple(M, vector<double>(P));
    double elapsed = simple_multiplication<double>(A, B, res_simple, M, N, P);
    cout << "Простое умножение заняло: " << elapsed << " секунд.\n";
    
    vector<vector<double>> res_eigen(M, vector<double>(P));
    double elapsed2 = eigen_multiplication<double>(A, B, res_eigen);
    cout << "Умножение через Eigen заняло: " << elapsed2  << " секунд.\n";

    vector<vector<double>> res_threaded(M, vector<double>(P));
    double elapsed3 = threaded_multiplication<double>(A, B, res_threaded, M, N, P);
    cout << "Многопоточное умножение заняло: " << elapsed3 << " секунд.\n";

    bool areEqual1 = compareMatrices(res_simple, res_threaded);
    bool areEqual2 = compareMatrices(res_simple, res_eigen);
    bool areEqual = areEqual1 & areEqual2;
    cout << "Матрицы равны: " << areEqual << endl;

    return 0;
}