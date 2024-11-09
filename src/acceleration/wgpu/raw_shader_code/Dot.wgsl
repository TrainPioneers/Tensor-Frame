@group(0) @binding(0) var<storage, read> vec_a : vec3<f32>;
@group(0) @binding(1) var<storage, read> vec_b : vec3<f32>;
@group(0) @binding(2) var<storage, write> dot_result : f32;

@compute @workgroup_size(1)
fn main() {
    // Calculate the dot product
    dot_result = dot(vec_a, vec_b);
}
