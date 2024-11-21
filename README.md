This is still alpha version, I will post it on crates.io soon

# Tensor-Frame

**Tensor-Frame** is a highly flexible and "efficient" library designed for multidimensional array (tensor) computations.
It supports operations on both CPU and GPU hardware.

## Hardware Support

This framework leverages the following libraries, optimized for different hardware, listed from fastest to slowest:<br>
*(Performance is hardware-dependent; rankings are relative.)*

- **cust** - Low-level Rust bindings for CUDA, enabling GPU acceleration on NVIDIA GPUs (requires an NVIDIA GPU and CUDA toolkit installed).
- **HIP** - Low-level framework for GPU acceleration, originally written in C/C++, designed for AMD GPUs with ROCm support (requires an AMD GPU and ROCm installation).
- **OpenCL** - Cross-platform parallel computing framework that works on GPUs, CPUs, and other hardware (requires OpenCL drivers and runtime installation).
- **WGPU** - High-level graphics and computation API in Rust built on Vulkan, Metal, DX12, and OpenGL, offering cross-platform GPU acceleration without requiring vendor-specific SDKs (only requires a GPU).
- **rayon** - Data-parallelism library for Rust that enables easy, safe, and efficient parallel computations on multi-core CPUs.

**Note: In many cases, Rayon outperforms wgpu due to GPU setup overhead (I hope to fix that in 0.1.2)**

## Docs
**As of 11/20/2024 there is no stable release on Cargo.io as 0.1.1 is still not finished**

Add Tensor-Frame to your project
`cargo add tensor_frame`
or
`tensor_frame = "0.1.1"`

Creating a simple tensor:
```rust
use tensor_frame::prelude::*;
// As of 11/20/2024 you can use
// use tensor_frame::Tensor;
// unless you are using vector math functions

fn main() {
  // Default type is i32
  let t1 = Tensor::ones(vec![100]); // shape of 100
  // also  Tensor::zeros(vec![100]);
  let t2 = Tensor::from_vec(vec![9, 100], vec![100]); // tensor with 100 elements and a shape of 100
  let t3 = Tensor::from_vec(vec![8, 100], vec![2,50]); // tensor with 100 elements and a shape of 2,50
  let t4 = &t1 + &t2; // this automatically runs it on avaliable hardware
  let t5 = &t1 + &t3; // it dosen't matter that the shape is different, just that the element amount is the same
  // but it does return the first shape so t4 is shape 100 not 2,50
  let t6 = &t3 + &t1; // shape is 2,50 not 100
  // We borrow the values because Copy is not implemented for Tensor. You can also use .clone()
  let t7 = t1.clone() + t2.clone(); // But this has more boilerplate
  let placeholder = Tensor::new();

  // explicit type
  let t8 = Tensor::<f64>::from_vec(vec![9, 100], vec![100]);
  // or
  let t9: Tensor<f64> = Tensor::from_vec(vec![9, 100], vec![100]);
  // or 
  let t10 = Tensor::from_vec(vec![9.0, 100], vec![100]); // this does not work on ::ones ::zeros
  // They all create a f64 Tensor
  // default is i32 if int is inputed and f64 if float is inputed

  // Comparing Tensors
  let same: bool = t8 == t9; // true
  let not_same: bool = t4 == t5; // false because different shapes
  let dif_vals: bool = t1 == t2; // false same shape different values 

  let v = run_vector_operation(vec![7, 100], vec![6, 100], RunOperation::Add); // only if you imported prelude
  // adds the vectors as result[i] = a[i] + b[i]
}
```
The Tensor struct supports for the usage of `i32`,`i64`,`f32`,`f64` (Unless you are using WGPU because it only supports `f32`)
<br><br>
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
