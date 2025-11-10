#ifndef KERNEL_LAUNCHER
#define KERNEL_LAUNCHER

#define BLOCK_DIM 16
#include <cuda_runtime.h>

template<typename KernelT>
double multiplyMatrices(
    KernelT multiplicationKernel,
    vector<vector<float>> &A, vector<vector<float>> &B, vector<vector<float>> &C
){
    int matrix_size = A.size();

    vector<float> flat_A = flattenMatrix(A);
    vector<float> flat_B = flattenMatrix(B);

    float *A_device, *B_device, *res_device;
    cudaMalloc(&A_device, matrix_size * matrix_size * sizeof(float));
    cudaMalloc(&B_device, matrix_size * matrix_size * sizeof(float));
    cudaMalloc(&res_device, matrix_size * matrix_size * sizeof(float));

    cudaMemcpy(A_device, flat_A.data(), matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, flat_B.data(), matrix_size * matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    // Определяем размер блока и сетки
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((matrix_size + blockSize.x - 1) / blockSize.x, (matrix_size + blockSize.y - 1) / blockSize.y);

    // Запускаем ядро
    auto start_time = chrono::high_resolution_clock::now();
    multiplicationKernel<<<gridSize, blockSize>>>(A_device, B_device, res_device, matrix_size);
    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;


    cudaDeviceSynchronize();
    vector<float> result(matrix_size * matrix_size);
    cudaMemcpy(result.data(), res_device, matrix_size * matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < matrix_size; ++i) {
        for(int j = 0; j < matrix_size; ++j) {
            C[i][j] = result[i * matrix_size + j];
        }
    }

    // Освобождаем память устройства
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(res_device);

    return elapsed.count();
}

#endif