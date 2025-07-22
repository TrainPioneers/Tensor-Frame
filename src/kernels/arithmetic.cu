// Arithmetic operations kernels
extern "C" {

// Element-wise addition
__global__ void add_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// Element-wise subtraction
__global__ void sub_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

// Element-wise multiplication
__global__ void mul_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

// Element-wise division with division by zero check
__global__ void div_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (b[idx] == 0.0f) {
            result[idx] = INFINITY; // Set to infinity for division by zero
        } else {
            result[idx] = a[idx] / b[idx];
        }
    }
}

} // extern "C"