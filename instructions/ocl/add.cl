__kernel void add(
    __global const float* a,       // input vector a
    __global const float* b,       // input vector b
    __global float* result,        // output vector result
    const int num_elements         // number of elements in vectors
) {
    int i = get_global_id(0);

    if (i < num_elements) {
        result[i] = a[i] + b[i];
    }
}
