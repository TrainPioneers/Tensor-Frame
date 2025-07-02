# Tensor Frame

Tensor Frame is a high-performance, PyTorch-like tensor library for Rust that supports multiple computational backends including CPU (with Rayon), WGPU (for GPU compute), and CUDA.

## Features

- **Multiple Backends**: Automatic backend selection with fallback support
  - CPU backend with Rayon for parallel processing
  - WGPU backend for cross-platform GPU computing
  - CUDA backend for NVIDIA GPU acceleration
- **PyTorch-like API**: Familiar tensor operations and broadcasting
- **Dynamic Tensors**: Runtime shape and type flexibility
- **Full Broadcasting Support**: NumPy-style automatic shape broadcasting for all arithmetic operations (+, -, *, /)
- **Zero-Copy Operations**: Efficient memory management where possible
- **Feature Flags**: Optional backends via Cargo features

## Quick Example

```rust
use tensor_frame::Tensor;

// Create tensors (automatically uses the best available backend)
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let b = Tensor::from_vec(vec![10.0, 20.0], vec![2, 1])?;

// Perform operations with automatic broadcasting
let c = (a + b)?;  // Broadcasting: [2,2] + [2,1] -> [2,2]
println!("Result: {:?}", c.to_vec()?); // [11.0, 12.0, 23.0, 24.0]

// All operations support broadcasting
let scalar = Tensor::from_vec(vec![2.0], vec![])?;
let scaled = (c / scalar)?;  // Divide by scalar
let sum = scaled.sum(None)?; // Sum all elements

println!("Sum: {:?}", sum.to_vec()?);
```

## Backend Priority

By default, Tensor Frame will attempt to use backends in this order:
1. CUDA (if available and feature enabled)
2. WGPU (if available and feature enabled)  
3. CPU (always available)

You can also explicitly specify a backend or create custom backend implementations.