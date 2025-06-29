# Operations Reference

This page provides detailed specifications for all tensor operations in Tensor Frame.

## Arithmetic Operations

### Element-wise Binary Operations

All element-wise operations support automatic broadcasting.

#### Addition (`+`)

```rust
fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>
```

Computes element-wise addition: `output[i] = lhs[i] + rhs[i]`

**Broadcasting**: Yes  
**Supported shapes**: Any compatible shapes  
**Error conditions**: Shape incompatibility

```rust
let a = Tensor::ones(vec![2, 3])?;
let b = Tensor::ones(vec![2, 3])?;
let c = a + b;  // All elements = 2.0
```

#### Subtraction (`-`)

```rust
fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>
```

Computes element-wise subtraction: `output[i] = lhs[i] - rhs[i]`

**Broadcasting**: Yes  
**Supported shapes**: Any compatible shapes  
**Error conditions**: Shape incompatibility

#### Multiplication (`*`)

```rust
fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>
```

Computes element-wise multiplication: `output[i] = lhs[i] * rhs[i]`

**Note**: This is NOT matrix multiplication. Use `matmul()` for matrix operations.

**Broadcasting**: Yes  
**Supported shapes**: Any compatible shapes  
**Error conditions**: Shape incompatibility

#### Division (`/`)

```rust
fn div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>
```

Computes element-wise division: `output[i] = lhs[i] / rhs[i]`

**Broadcasting**: Yes  
**Supported shapes**: Any compatible shapes  
**Error conditions**: Shape incompatibility, division by zero

## Matrix Operations

### Matrix Multiplication

```rust
fn matmul(&self, other: &Tensor) -> Result<Tensor>
```

Performs matrix multiplication between 2D tensors.

**Supported shapes**: 
- `[m, k] × [k, n] -> [m, n]`
- Both tensors must be 2D

**Error conditions**: 
- Non-2D tensors
- Incompatible inner dimensions
- Backend not supporting matmul

```rust
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
let c = a.matmul(&b)?;
// Result: [[19.0, 22.0], [43.0, 50.0]]
```

## Reduction Operations

### Sum

```rust
fn sum(&self, axis: Option<usize>) -> Result<Tensor>
```

Computes sum along specified axis or all elements.

**Parameters**:
- `axis: None` - Sum all elements, return scalar tensor
- `axis: Some(i)` - Sum along axis `i`, reduce that dimension

**Supported shapes**: Any  
**Error conditions**: Invalid axis index

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

// Sum all elements
let total = tensor.sum(None)?;          // Result: [10.0] (scalar)

// Sum along axis 0 (rows)
let col_sums = tensor.sum(Some(0))?;    // Result: [4.0, 6.0]

// Sum along axis 1 (columns)  
let row_sums = tensor.sum(Some(1))?;    // Result: [3.0, 7.0]
```

### Mean

```rust
fn mean(&self, axis: Option<usize>) -> Result<Tensor>
```

Computes arithmetic mean along specified axis or all elements.

**Parameters**:
- `axis: None` - Mean of all elements, return scalar tensor
- `axis: Some(i)` - Mean along axis `i`, reduce that dimension

**Supported shapes**: Any  
**Error conditions**: Invalid axis index

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;

// Mean of all elements
let average = tensor.mean(None)?;       // Result: [2.5] (scalar)

// Mean along axis 0
let col_means = tensor.mean(Some(0))?;  // Result: [2.0, 3.0]
```

## Shape Manipulation

### Reshape

```rust
fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor>
```

Changes tensor shape while preserving total number of elements.

**Requirements**: 
- Product of new_shape must equal `self.numel()`
- New shape cannot have zero dimensions

**Error conditions**: 
- Incompatible total elements
- Invalid shape dimensions

```rust
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
let reshaped = tensor.reshape(vec![3, 2])?;  // 2×3 -> 3×2
let flattened = tensor.reshape(vec![6])?;    // 2×3 -> 6×1
```

### Transpose

```rust
fn transpose(&self) -> Result<Tensor>
```

Transposes a 2D tensor (swaps dimensions).

**Requirements**: Tensor must be exactly 2D  
**Error conditions**: Non-2D tensor

```rust
let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
let transposed = matrix.transpose()?;
// [[1,2],[3,4]] -> [[1,3],[2,4]]
```

### Squeeze

```rust
fn squeeze(&self, dim: Option<usize>) -> Result<Tensor>
```

Removes dimensions of size 1.

**Parameters**:
- `dim: None` - Remove all dimensions of size 1
- `dim: Some(i)` - Remove dimension `i` only if it has size 1

**Error conditions**: 
- Invalid dimension index
- Trying to squeeze dimension with size > 1

```rust
let tensor = Tensor::ones(vec![1, 3, 1, 2])?;  // Shape: [1, 3, 1, 2]
let squeezed = tensor.squeeze(None)?;          // Shape: [3, 2]
let partial = tensor.squeeze(Some(0))?;        // Shape: [3, 1, 2]
```

### Unsqueeze

```rust
fn unsqueeze(&self, dim: usize) -> Result<Tensor>
```

Adds a dimension of size 1 at the specified position.

**Parameters**:
- `dim` - Position to insert new dimension (0 to ndim inclusive)

**Error conditions**: Invalid dimension index (> ndim)

```rust
let tensor = Tensor::ones(vec![3, 2])?;      // Shape: [3, 2]
let unsqueezed = tensor.unsqueeze(0)?;       // Shape: [1, 3, 2]
let middle = tensor.unsqueeze(1)?;           // Shape: [3, 1, 2]
let end = tensor.unsqueeze(2)?;              // Shape: [3, 2, 1]
```

## Broadcasting Rules

Tensor Frame follows NumPy/PyTorch broadcasting conventions:

### Alignment
Shapes are aligned from the rightmost dimension:
```
Tensor A: [3, 1, 4]
Tensor B:    [2, 4]
Result:   [3, 2, 4]
```

### Size 1 Expansion
Dimensions of size 1 are expanded to match:
```
Tensor A: [3, 1, 4]
Tensor B: [3, 2, 1]  
Result:   [3, 2, 4]
```

### Missing Dimensions
Missing leading dimensions are treated as size 1:
```
Tensor A: [5, 3, 2]
Tensor B:    [3, 2]
Result:   [5, 3, 2]
```

### Incompatible Shapes
These shapes cannot be broadcast:
```
Tensor A: [3, 4]
Tensor B: [2, 4]  # Error: 3 and 2 cannot be broadcast
```

## Performance Notes

### Operation Fusion
- Operations on the same backend avoid intermediate allocations when possible
- Sequential reductions can be fused into single kernel calls

### Memory Layout
- All tensors use row-major (C-style) memory layout
- Reshape operations are zero-copy when layout permits
- Transpose creates new memory layout

### Backend-Specific Optimizations
- **CPU**: Uses Rayon for parallel element-wise operations
- **WGPU**: Utilizes compute shaders for parallel GPU execution
- **CUDA**: Leverages cuBLAS for matrix operations and custom kernels for element-wise ops

### Broadcasting Performance
- Zero-copy broadcasting when one tensor has size-1 dimensions
- Explicit memory expansion fallback for complex broadcasting patterns
- GPU backends optimize broadcasting in compute shaders