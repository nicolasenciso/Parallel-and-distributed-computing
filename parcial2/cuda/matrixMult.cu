#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;
#define BLOCK_SIZE 8

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if ( col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }

        c[row * k + col] = sum;
    }
}



void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            int tmp = 0.0;

            for (int h = 0; h < n; ++h) {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }

            h_result[i * k + j] = tmp;
        }
    }
}


int main(int argc, char const *argv[]) {
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    n = atoi(argv[1]);
    m = n;
    k = n;
    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m * n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n * k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m * k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m * k);
    
 auto startClock = chrono::steady_clock::now();
    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    // Allocate memory space on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m * n);
    cudaMalloc((void **) &d_b, sizeof(int)*n * k);
    cudaMalloc((void **) &d_c, sizeof(int)*m * k);
    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n * k, cudaMemcpyHostToDevice);
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE * atoi(argv[2]), BLOCK_SIZE * atoi(argv[2]));
    gpu_matrix_mult <<< dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    // Transefr results from device to host
    cudaMemcpy(h_c, d_c, sizeof(int)*m * k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    //  printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    // start the CPU version
    cudaEventRecord(start, 0);
    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    //printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);
    // validate results computed by GPU
    int all_ok = 1;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if (h_cc[i * k + j] != h_c[i * k + j]) {
                all_ok = 0;
            }
        }

        //printf("\n");
    }

    // roughly compute speedup
    if (all_ok) {
        //  printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
        printf("%d\t%d\t%f\n", n, atoi(argv[2]), cpu_elapsed_time_ms / gpu_elapsed_time_ms);

    } else {
        printf("incorrect results\n");
    }
    //end clock timing
    auto endClock = chrono::steady_clock::now();
    auto finalClock = endClock - startClock;
    cout<< n << "," << chrono::duration <double, milli> (finalClock).count()<<endl;
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
