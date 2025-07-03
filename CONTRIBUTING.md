# Contributing to Tensor Frame

We welcome contributions to Tensor Frame! This guide will help you get started with contributing to the project.

## Getting Started

### Development Setup

1. **Clone the repository**:
```bash
git clone https://github.com/TrainPioneers/Tensor-Frame.git
cd Tensor-Frame
```

2. **Install Rust** (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

3. **Install development dependencies**:
```bash
# For benchmarking
cargo install criterion

# For code formatting
rustup component add rustfmt

# For linting
rustup component add clippy
```

4. **Build and test**:
```bash
# Build with all features
cargo build --all-features

# Run tests using make
make test

# Run with specific backend
cargo test --features wgpu
cargo test --features cuda
```

## Development Workflow

### Building the Project

```bash
# Quick compilation check
cargo check

# Build with specific backends
cargo build --features wgpu
cargo build --features cuda
cargo build --all-features

# Release build
cargo build --release --all-features
```

### Running Tests

```bash
# Run all tests using make
make test

# Test individual backends
make test-cpu
make test-wgpu  
make test-cuda

# Test with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_tensor_creation
```

### Code Formatting and Linting

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check

# Run clippy lints
cargo clippy

# Run clippy with all features
cargo clippy --all-features
```

### Documentation

```bash
# Generate API documentation
cargo doc --open

# Build the book
cd docs
mdbook build

# Serve book locally
mdbook serve
```

## Contribution Guidelines

### Code Style

- **Formatting**: Use `cargo fmt` for consistent formatting
- **Linting**: Address all `cargo clippy` warnings
- **Naming**: Use descriptive names following Rust conventions
- **Comments**: Document public APIs and complex algorithms
- **Error Handling**: Use proper `Result` types and meaningful error messages

### Testing

All contributions must include appropriate tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_feature() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        let result = tensor.new_operation().unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_error_handling() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        let result = tensor.invalid_operation();
        assert!(result.is_err());
    }
}
```

### Documentation Requirements

- **Public APIs**: All public functions, structs, and traits must have documentation
- **Examples**: Include usage examples in documentation
- **Error Cases**: Document when functions return errors

```rust
/// Creates a new tensor filled with zeros.
///
/// # Arguments
/// * `shape` - The dimensions of the tensor
///
/// # Returns
/// A new tensor filled with zeros, or an error if the shape is invalid.
///
/// # Examples
/// ```
/// use tensor_frame::Tensor;
/// 
/// let tensor = Tensor::zeros(vec![2, 3])?;
/// assert_eq!(tensor.numel(), 6);
/// # Ok::<(), tensor_frame::TensorError>(())
/// ```
///
/// # Errors
/// Returns `TensorError::InvalidShape` if any dimension is zero.
pub fn zeros(shape: Vec<usize>) -> Result<Self> {
    // Implementation
}
```

## Types of Contributions

### Bug Fixes

1. **Report the issue**: Create a GitHub issue with:
   - Clear reproduction steps
   - Expected vs actual behavior
   - Environment details (OS, Rust version, GPU info)
   - Minimal code example

2. **Fix the bug**:
   - Create a focused fix addressing the specific issue
   - Add regression tests to prevent recurrence
   - Update documentation if the bug was in documented behavior

### New Features

Before implementing new features:

1. **Discuss the feature**: Open a GitHub issue to discuss:
   - Use case and motivation
   - Proposed API design
   - Implementation approach
   - Performance implications

2. **Implementation guidelines**:
   - Follow existing patterns and conventions
   - Implement for all relevant backends
   - Add comprehensive tests
   - Update documentation and examples

#### Backend Implementation

New operations should be implemented across all backends:

```rust
// src/backend/mod.rs
pub trait Backend {
    // Add new operation to trait
    fn new_operation(&self, input: &Storage) -> Result<Storage>;
}

// Implement in CPU, WGPU, and CUDA backends
```

## Current Architecture

### Core Components

- **Tensor**: Main tensor struct with shape and storage
- **Backends**: CPU (Rayon), WGPU (cross-platform GPU), CUDA (NVIDIA GPU)
- **Operations**: Arithmetic, reduction, reshape, transpose
- **Broadcasting**: Automatic shape compatibility (partial implementation)

### Backend Priority

Tensor Frame automatically selects backends in this order:
1. **CUDA** (if available and compiled)
2. **WGPU** (if available and compiled)
3. **CPU** (always available as fallback)

### Current Features

- ✅ Tensor creation (zeros, ones, from_vec)
- ✅ Basic arithmetic (+, -, *, /)
- ✅ Broadcasting (addition only)
- ✅ Reductions (sum, mean)
- ✅ Shape operations (reshape, transpose, squeeze, unsqueeze)
- ✅ Multiple backends with automatic selection

### Known Limitations

- Broadcasting only implemented for addition
- No matrix multiplication (not yet implemented)
- No advanced operations (convolution, etc.)

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
```bash
make test
```

2. **Check formatting and lints**:
```bash
cargo fmt --check
cargo clippy --all-features
```

3. **Update documentation if needed**:
```bash
cargo doc --all-features
```

### Pull Request Checklist

- [ ] Did I use the .githooks to see if my code passes tests?
- [ ] My code follows the project's code style
- [ ] I have added tests that prove my fix/feature works
- [ ] All tests pass locally
- [ ] I have updated documentation as needed
- [ ] My changes generate no new warnings

## Getting Help

If you need help contributing:

1. **Read existing code**: Look at similar implementations for patterns
2. **Check documentation**: API docs contain guidance
3. **Ask questions**: Open a GitHub issue or discussion
4. **Start small**: Begin with bug fixes or documentation improvements

Thank you for contributing to Tensor Frame!