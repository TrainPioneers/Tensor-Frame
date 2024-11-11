// kernel.cu
#include <cuda_runtime.h>
#include <math.h>

// Kernel function to add two vectors
__global__ void add(float *vec1, float *vec2, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec1[idx] + vec2[idx];
    }
}

// Kernel function to subtract two vectors
__global__ void sub(float *vec1, float *vec2, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec1[idx] - vec2[idx];
    }
}

// Kernel function to multiply two vectors
__global__ void mul(float *vec1, float *vec2, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec1[idx] * vec2[idx];
    }
}

// Kernel function to divide two vectors
__global__ void div(float *vec1, float *vec2, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (vec2[idx] != 0) {
            result[idx] = vec1[idx] / vec2[idx];
        } else {
            printf("Warning: Division by zero at index %d\n", idx);
        }
    }
}

// Kernel function to compute the dot product of two vectors
__global__ void dot(float *vec1, float *vec2, float *result, int n) {
    __shared__ float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;

    while (idx < n) {
        temp += vec1[idx] * vec2[idx];
        idx += blockDim.x;
    }

    cache[threadIdx.x] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *result = cache[0];
    }
}