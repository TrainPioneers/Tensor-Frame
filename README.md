This is still alpha version, I will post it on crates.io soon

# Tensor-Frame

**Tensor-Frame** is a highly flexible and "efficient" library designed for multidimensional array (tensor) computations.
It supports operations on both CPU and GPU hardware.

## Hardware Support

This framework leverages the following libraries, optimized for different hardware, listed from fastest to slowest:<br>
*(Performance is hardware-dependent; rankings are relative.)*

- **cust** - Low-level Rust bindings for CUDA, enabling GPU acceleration on NVIDIA GPUs (requires an NVIDIA GPU and CUDA toolkit installed).
- (Not available yet) **HIP** - Low-level framework for GPU acceleration, originally written in C/C++, designed for AMD GPUs with ROCm support (requires an AMD GPU and ROCm installation).
- (Not available yet) **OpenCL** - Cross-platform parallel computing framework that works on GPUs, CPUs, and other hardware (requires OpenCL drivers and runtime installation).
- (Not available yet) **WGPU** - High-level graphics and computation API in Rust built on Vulkan, Metal, DX12, and OpenGL, offering cross-platform GPU acceleration without requiring vendor-specific SDKs (only requires a GPU).
- **rayon** - Data-parallelism library for Rust that enables easy, safe, and efficient parallel computations on multi-core CPUs.

**Note: In many cases, Rayon outperforms wgpu due to GPU setup overhead (I hope to fix that in 0.1.2)**

## Docs
**As of 11/20/2024 there is no stable release on Cargo.io as 0.1.1 is still not finished**

### Importing and Setting Up
Add Tensor-Frame to your project
`cargo add tensor_frame`
or
`tensor_frame = "0.2.0"`

The `tensor_frame::prelude::*` module is imported to work with the Tensor struct and vector math functions.

```rust
fn main() {
    use tensor_frame::prelude::*;

    // Create a Tensor with a shape of 100 filled with ones.
    let t1 = Tensor::ones(vec![100]);
}
```

### Creating Tensors with Different Shapes
You can initialize tensors with varying shapes and dimensions using the `Tensor::from_vec` method.

```rust
fn main() {
    use tensor_frame::prelude::*;

    // A tensor with 100 elements and a shape of 100.
    let t1 = Tensor::from_vec(vec![9, 100], vec![100]);

    // A tensor with 100 elements and a shape of 2x50.
    let t2 = Tensor::from_vec(vec![8, 100], vec![2, 50]);
}
```

### Performing Operations on Tensors
Tensor operations like addition automatically utilize available hardware. The resulting shape depends on the first operand unless stated otherwise.

```rust
fn main() {
    use tensor_frame::prelude::*;

    let t1 = Tensor::ones(vec![100]);
    let t2 = Tensor::from_vec(vec![9, 100], vec![100]);
    let t3 = Tensor::from_vec(vec![8, 100], vec![2, 50]);

    let t4 = &t1 + &t2; // Resulting shape: 100
    let t5 = &t1 + &t3; // Shape remains 100 as t1 is the first operand.
    let t6 = &t3 + &t1; // Shape changes to 2x50 as t3 is the first operand.
}
```

### Using Explicit Types
You can specify the data type explicitly when creating tensors. The default type is `i32` for integers and `f64` for floats.

```rust
fn main() {
    use tensor_frame::prelude::*;

    // Explicitly set type to f64.
    let t1 = Tensor::<f64>::from_vec(vec![9, 100], vec![100]);

    // Alternative syntax using type annotation.
    let t2: Tensor<f64> = Tensor::from_vec(vec![9, 100], vec![100]);

    // Default type inferred as f64 for floats.
    let t3 = Tensor::from_vec(vec![9.0, 100], vec![100]);

    // An f32 Tensor
    let t4 = Tensor::<f32>::from_vec(vec![9, 100], vec![100]);

    // An i64 Tensor
    let t4 = Tensor::<i64>::from_vec(vec![9, 100], vec![100]);
}
```

### Comparing Tensors
Tensor comparison considers both shape and values.

```rust
fn main() {
    use tensor_frame::prelude::*;

    let t1 = Tensor::ones(vec![100]);
    let t2 = Tensor::ones(vec![100]);
    let t3 = Tensor::from_vec(vec![9, 100], vec![100]);
    let t4 = Tensor::from_vec(vec![8, 100], vec![2, 50]);

    let t1_equals_t2 = t1 == t2; // true
    let t1_equals_t3 = t1 == t3; // false
    let t2_equals_t4 = t2 == t4; // false
}
```

### Running Vector Operations
The `run_vector_operation` function can perform element-wise operations on vectors, such as addition.

```rust
fn main() {
    use tensor_frame::prelude::*;

    let v = run_vector_operation(vec![7, 100], vec![6, 100], RunOperation::Add);
    // Adds the vectors element-wise: result[i] = a[i] + b[i].
}
```

### Supported Data Types
The `Tensor` struct supports `i32`, `i64`, `f32`, and `f64`. Note that when using WGPU, only `f32` is supported.

### Possible Operands
RunOperation is defined as the following
```rust
#[derive(Debug, Hash, PartialEq, Eq)]
pub enum RunOperation {
    Add,
    Sub,
    Mul,
    Div,
}
```
<br><br>
I am planning to add customizations so you can define what you want to be true and what false, what is possible and not possible.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.
