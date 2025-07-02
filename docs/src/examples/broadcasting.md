# Broadcasting Examples

Broadcasting is one of the most powerful features in Tensor Frame, allowing operations between tensors of different shapes. This guide provides comprehensive examples of broadcasting patterns and best practices.

## Broadcasting Rules

Tensor Frame follows NumPy/PyTorch broadcasting rules:

1. **Alignment**: Shapes are compared element-wise from the trailing dimension
2. **Size 1 Expansion**: Dimensions of size 1 are expanded to match
3. **Missing Dimensions**: Missing leading dimensions are treated as size 1
4. **Compatibility**: Dimensions must be either equal, or one must be 1

## Basic Broadcasting Examples

### Scalar Broadcasting

```rust
use tensor_frame::{Tensor, Result};

fn scalar_broadcasting() -> Result<()> {
    // Create a base tensor
    let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2])?;
    println!("Original tensor:\n{}\n", tensor);
    
    // Scalar tensor for broadcasting
    let scalar = Tensor::from_vec(vec![2.0], vec![])?;
    
    // All operations support broadcasting
    let add_result = (tensor.clone() + scalar.clone())?;
    println!("Tensor + 2.0:\n{}\n", add_result);
    
    let sub_result = (tensor.clone() - scalar.clone())?;
    println!("Tensor - 2.0:\n{}\n", sub_result);
    
    let mul_result = (tensor.clone() * scalar.clone())?;
    println!("Tensor * 2.0:\n{}\n", mul_result);
    
    let div_result = (tensor.clone() / scalar.clone())?;
    println!("Tensor / 2.0:\n{}\n", div_result);
    
    Ok(())
}
```

### Vector Broadcasting

```rust
fn vector_broadcasting() -> Result<()> {
    // Matrix-vector operations
    let matrix = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;
    let vector = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3])?;
    
    println!("Matrix (2x3):\n{}\n", matrix);
    println!("Vector (3,):\n{}\n", vector);
    
    // All arithmetic operations support broadcasting
    let add_result = (matrix.clone() + vector.clone())?;
    println!("Matrix + Vector:\n{}\n", add_result);
    
    let mul_result = (matrix.clone() * vector.clone())?;
    println!("Matrix * Vector (element-wise):\n{}\n", mul_result);
    
    // Row vector broadcasting
    let row_vector = Tensor::from_vec(vec![100.0, 200.0], vec![2, 1])?;
    let row_add = (matrix.clone() + row_vector.clone())?;
    let row_sub = (matrix.clone() - row_vector.clone())?;
    println!("Matrix + Row Vector (2x1):\n{}\n", row_add);
    println!("Matrix - Row Vector (2x1):\n{}\n", row_sub);
    
    // Complex broadcasting example
    let a = Tensor::from_vec(vec![10.0, 20.0], vec![2, 1])?;
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3])?;
    let complex_result = (a / b)?;  // Broadcasting: [2,1] / [1,3] -> [2,3]
    println!("Complex broadcasting [2,1] / [1,3]:\n{}\n", complex_result);
    
    Ok(())
}
```

## Advanced Broadcasting Patterns

### Multi-dimensional Broadcasting

```rust
fn multidimensional_broadcasting() -> Result<()> {
    // 3D tensor broadcasting
    let tensor_3d = Tensor::ones(vec![2, 3, 4])?;     // Shape: [2, 3, 4]
    let tensor_2d = Tensor::ones(vec![3, 4])?;        // Shape: [3, 4]
    let tensor_1d = Tensor::ones(vec![4])?;           // Shape: [4]
    
    println!("3D tensor shape: {:?}", tensor_3d.shape().dims());
    println!("2D tensor shape: {:?}", tensor_2d.shape().dims());
    println!("1D tensor shape: {:?}", tensor_1d.shape().dims());
    
    // 3D + 2D broadcasting: [2,3,4] + [3,4] -> [2,3,4]
    let result_3d_2d = &tensor_3d + &tensor_2d;
    println!("3D + 2D result shape: {:?}", result_3d_2d.shape().dims());
    
    // 3D + 1D broadcasting: [2,3,4] + [4] -> [2,3,4]
    let result_3d_1d = &tensor_3d + &tensor_1d;
    println!("3D + 1D result shape: {:?}", result_3d_1d.shape().dims());
    
    // Complex multi-dimensional broadcasting
    let a = Tensor::ones(vec![1, 3, 1])?;             // Shape: [1, 3, 1]
    let b = Tensor::ones(vec![2, 1, 4])?;             // Shape: [2, 1, 4]
    let complex_result = &a + &b;                     // Result: [2, 3, 4]
    
    println!("Complex broadcasting:");
    println!("  A shape: {:?}", a.shape().dims());
    println!("  B shape: {:?}", b.shape().dims());
    println!("  Result shape: {:?}", complex_result.shape().dims());
    
    Ok(())
}
```

### Broadcasting with Size-1 Dimensions

```rust
fn size_one_broadcasting() -> Result<()> {
    // Different ways to create broadcastable tensors
    let base = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    )?;
    
    // Row broadcasting (1 x N)
    let row_broadcast = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![1, 3])?;
    let row_result = &base + &row_broadcast;
    println!("Row broadcasting [2,3] + [1,3]:\n{}\n", row_result);
    
    // Column broadcasting (N x 1)
    let col_broadcast = Tensor::from_vec(vec![100.0, 200.0], vec![2, 1])?;
    let col_result = &base + &col_broadcast;
    println!("Column broadcasting [2,3] + [2,1]:\n{}\n", col_result);
    
    // Both dimensions broadcast (1 x 1)
    let scalar_as_tensor = Tensor::from_vec(vec![1000.0], vec![1, 1])?;
    let scalar_result = &base + &scalar_as_tensor;
    println!("Scalar broadcasting [2,3] + [1,1]:\n{}\n", scalar_result);
    
    Ok(())
}
```

## Broadcasting in Practice

### Machine Learning Patterns

```rust
fn ml_broadcasting_patterns() -> Result<()> {
    // Batch normalization pattern
    let batch_data = Tensor::ones(vec![32, 128])?;    // 32 samples, 128 features
    let mean = Tensor::zeros(vec![128])?;             // Feature means
    let std = Tensor::ones(vec![128])?;               // Feature standard deviations
    
    // Normalize: (x - mean) / std
    let normalized = (&batch_data - &mean) / &std;
    println!("Batch normalization result shape: {:?}", normalized.shape().dims());
    
    // Bias addition pattern  
    let linear_output = Tensor::ones(vec![32, 10])?;  // Batch size 32, 10 classes
    let bias = Tensor::from_vec(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        vec![10]
    )?;
    
    let biased_output = &linear_output + &bias;
    println!("Bias addition result shape: {:?}", biased_output.shape().dims());
    
    // Attention score broadcasting
    let queries = Tensor::ones(vec![32, 8, 64])?;     // [batch, heads, dim]
    let attention_weights = Tensor::ones(vec![32, 8, 1])?;  // [batch, heads, 1]
    
    let weighted_queries = &queries * &attention_weights;
    println!("Attention weighting result shape: {:?}", weighted_queries.shape().dims());
    
    Ok(())
}
```

### Image Processing Patterns

```rust
fn image_broadcasting_patterns() -> Result<()> {
    // Image batch processing
    let images = Tensor::ones(vec![4, 3, 224, 224])?;  // [batch, channels, height, width]
    
    // Channel-wise normalization
    let channel_mean = Tensor::from_vec(
        vec![0.485, 0.456, 0.406],  // ImageNet means
        vec![1, 3, 1, 1]
    )?;
    let channel_std = Tensor::from_vec(
        vec![0.229, 0.224, 0.225],  // ImageNet stds
        vec![1, 3, 1, 1]
    )?;
    
    let normalized_images = (&images - &channel_mean) / &channel_std;
    println!("Image normalization result shape: {:?}", normalized_images.shape().dims());
    
    // Pixel-wise operations
    let brightness_adjustment = Tensor::from_vec(vec![0.1], vec![1, 1, 1, 1])?;
    let brightened = &images + &brightness_adjustment;
    println!("Brightness adjustment result shape: {:?}", brightened.shape().dims());
    
    Ok(())
}
```

## Performance Considerations

### Efficient Broadcasting

```rust
use std::time::Instant;

fn broadcasting_performance() -> Result<()> {
    // Efficient: Broadcasting avoids large intermediate tensors
    let large_matrix = Tensor::ones(vec![1000, 1000])?;
    let small_vector = Tensor::ones(vec![1000])?;
    
    let start = Instant::now();
    let efficient_result = &large_matrix + &small_vector;  // Broadcasting
    let efficient_time = start.elapsed();
    
    println!("Efficient broadcasting: {:?}", efficient_time);
    
    // Less efficient: Explicit expansion (don't do this!)
    let start = Instant::now();
    let expanded_vector = small_vector.reshape(vec![1, 1000])?;
    // Note: This would need manual tiling which isn't implemented
    // let manual_result = &large_matrix + &expanded_vector;
    let manual_time = start.elapsed();
    
    println!("Manual expansion overhead: {:?}", manual_time);
    
    Ok(())
}
```

### Memory-Efficient Patterns

```rust
fn memory_efficient_broadcasting() -> Result<()> {
    // Good: Broadcasting reuses memory
    let data = Tensor::ones(vec![1000, 500])?;
    let scale_factor = Tensor::from_vec(vec![2.0], vec![1])?;
    
    let scaled = &data * &scale_factor;  // Memory efficient
    
    // Avoid: Creating large intermediate tensors
    // let large_scale = scale_factor.broadcast_to(vec![1000, 500])?;  // Wasteful
    // let scaled = &data * &large_scale;
    
    println!("Memory-efficient scaling completed");
    
    Ok(())
}
```

## Common Broadcasting Errors

### Shape Incompatibility

```rust
fn broadcasting_errors() -> Result<()> {
    // These will fail - incompatible shapes
    let a = Tensor::ones(vec![3, 4])?;
    let b = Tensor::ones(vec![2, 4])?;  // Different first dimension, not 1
    
    match &a + &b {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error - incompatible shapes: {}", e),
    }
    
    // These will work - compatible shapes
    let c = Tensor::ones(vec![1, 4])?;  // First dimension is 1
    let success = &a + &c;
    println!("Compatible shapes work: {:?}", success.shape().dims());
    
    Ok(())
}
```

## Broadcasting Visualization

### Understanding Shape Alignment

```rust
fn visualize_broadcasting() -> Result<()> {
    println!("Broadcasting visualization:");
    println!();
    
    // Example 1: [2, 3] + [3]
    println!("Example 1: [2, 3] + [3]");
    println!("  A: [2, 3]");  
    println!("  B:    [3]  ->  [1, 3]  (implicit leading 1)");
    println!("  Result: [2, 3]");
    println!();
    
    // Example 2: [4, 1, 5] + [3, 5]
    println!("Example 2: [4, 1, 5] + [3, 5]");
    println!("  A: [4, 1, 5]");
    println!("  B:    [3, 5]  ->  [1, 3, 5]  (implicit leading 1)");
    println!("  Result: [4, 3, 5]  (1 broadcasts to 3, 4)");
    println!();
    
    // Example 3: Incompatible
    println!("Example 3: [3, 4] + [2, 4] - INCOMPATIBLE");
    println!("  A: [3, 4]");
    println!("  B: [2, 4]");
    println!("  Error: 3 and 2 cannot broadcast (neither is 1)");
    println!();
    
    Ok(())
}
```

## Best Practices

### 1. Design for Broadcasting
```rust
// Good: Design tensors with broadcasting in mind
let batch_size = 32;
let features = 128;

let data = Tensor::ones(vec![batch_size, features])?;
let weights = Tensor::ones(vec![features])?;          // Broadcastable
let bias = Tensor::ones(vec![features])?;             // Broadcastable

let output = (&data * &weights) + &bias;  // Clean broadcasting
```

### 2. Use Explicit Shapes
```rust
// Better: Be explicit about intended broadcasting
let matrix = Tensor::ones(vec![10, 20])?;
let row_vector = Tensor::ones(vec![1, 20])?;    // Explicit [1, 20]
let col_vector = Tensor::ones(vec![10, 1])?;    // Explicit [10, 1]

let row_broadcast = &matrix + &row_vector;
let col_broadcast = &matrix + &col_vector;
```

### 3. Document Broadcasting Intent
```rust
/// Applies per-channel normalization to image batch
/// 
/// # Arguments
/// * `images` - Shape [batch, channels, height, width]
/// * `channel_stats` - Shape [1, channels, 1, 1] for broadcasting
fn normalize_images(images: &Tensor, channel_stats: &Tensor) -> Result<Tensor> {
    // Broadcasting: [B,C,H,W] - [1,C,1,1] -> [B,C,H,W]
    images - channel_stats
}
```

### 4. Validate Shapes Early
```rust
fn safe_broadcast_operation(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check compatibility before expensive operations
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();
    
    // Custom validation logic here
    if !shapes_are_broadcastable(a_shape, b_shape) {
        return Err(TensorError::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }
    
    // Proceed with operation
    a + b
}

fn shapes_are_broadcastable(a: &[usize], b: &[usize]) -> bool {
    let max_len = a.len().max(b.len());
    
    for i in 0..max_len {
        let a_dim = a.get(a.len().saturating_sub(max_len - i)).unwrap_or(&1);
        let b_dim = b.get(b.len().saturating_sub(max_len - i)).unwrap_or(&1);
        
        if *a_dim != *b_dim && *a_dim != 1 && *b_dim != 1 {
            return false;
        }
    }
    true
}
```

## Next Steps

After mastering broadcasting:
- [Custom Backends](./custom-backends.md) - Optimize broadcasting for different backends
- [Performance Guide](../performance.md) - Advanced broadcasting optimization
- [API Reference](../api/operations.md) - Detailed operation specifications