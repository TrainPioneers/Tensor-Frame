__kernel void div(
    __global const float* a,       // input vector a
    __global const float* b,       // input vector b
    __global float* result,        // output vector result
    const int num_elements         // number of elements in vectors
) {
    int i = get_global_id(0);

    if (i < num_elements) {
        // Handle divide by zero
        if (b[i] != 0.0f) {
            result[i] = a[i] / b[i];
        } else {
            result[i] = 0.0f; // Can be NaN or another error value
        }
    }
}
