# Tensor Frame

[![Crates.io](https://img.shields.io/crates/v/tensor_frame)](https://crates.io/crates/tensor_frame)
[![Documentation](https://docs.rs/tensor_frame/badge.svg)](https://docs.rs/tensor_frame)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A high-performance, PyTorch-like tensor library for Rust with support for multiple computational backends.

## Features

- 🚀 **Multiple Backends**: CPU (Rayon), WGPU, and CUDA support
- 🔄 **Automatic Backend Selection**: Falls back to best available backend
- 📐 **Full Broadcasting**: NumPy/PyTorch-style automatic broadcasting for all arithmetic operations
- 🎯 **Type Safety**: Rust's type system for memory safety
- ⚡ **Zero-Copy Operations**: Efficient memory management
- 🎛️ **Feature Flags**: Optional dependencies for different backends

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tensor_frame = "0.0.1-alpha"

# For GPU support
tensor_frame = { version = "0.0.1-alpha", features = ["wgpu"] }
```

Basic usage:

```rust
use tensor_frame::Tensor;

// Create tensors (automatically uses best backend)
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let b = Tensor::from_vec(vec![10.0, 20.0], vec![2, 1])?;

// All operations support broadcasting: +, -, *, /
let c = (a + b)?;  // Broadcasting: [2,2] + [2,1] -> [2,2]
let d = (c * b)?;  // Element-wise multiplication with broadcasting
let sum = d.sum(None)?;

println!("Result: {:?}", sum.to_vec()?);
```

## Backends

### CPU Backend (Default)
- Uses Rayon for parallel computation
- Always available
- Good for small to medium tensors

### WGPU Backend
- Cross-platform GPU compute
- Supports Metal, Vulkan, DX12, OpenGL
- Enable with `features = ["wgpu"]`

### CUDA Backend  
- NVIDIA GPU acceleration
- Enable with `features = ["cuda"]`
- Requires CUDA toolkit

## Documentation

- 📖 [**Complete Guide**](https://trainpioneers.github.io/Tensor-Frame/) - Comprehensive documentation with tutorials
- 🚀 [**Getting Started**](https://trainpioneers.github.io/Tensor-Frame/getting-started.html) - Quick start guide  
- 📚 [**API Reference**](https://docs.rs/tensor_frame) - Detailed API documentation
- 💡 [**Examples**](https://trainpioneers.github.io/Tensor-Frame/examples/) - Practical examples and tutorials
- ⚡ [**Performance Guide**](https://trainpioneers.github.io/Tensor-Frame/performance.html) - Optimization tips and benchmarks
- 🔧 [**Backend Guides**](https://trainpioneers.github.io/Tensor-Frame/backends/) - CPU, WGPU, and CUDA backend details

## Examples

See the [examples](examples/) directory for more detailed usage:

- [Basic Operations](examples/basic_operations.rs)
- [Broadcasting](examples/broadcasting.rs)
- [Backend Selection](examples/backend_selection.rs)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.