@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input_a)) {
        return;
    }
    
    let x = input_a[index];
    let y = input_b[index];
    
    // Simple division - let the GPU hardware handle IEEE 754 behavior
    // This will naturally produce NaN, +inf, and -inf as expected
    output[index] = x / y;
}