// kernel.cu
#include <cuda_runtime.h>
#include <math.h>

// Kernel function to add two vectors
__global__ void add(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// Kernel function to subtract two vectors
__global__ void sub(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

// Kernel function to multiply two vectors
__global__ void mul(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

// Kernel function to divide two vectors
__global__ void div(float *a, float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (b[idx] != 0) {
            result[idx] = a[idx] / b[idx];
        } else {
            printf("Warning: Division by zero at index %d\n", idx);
        }
    }
}
