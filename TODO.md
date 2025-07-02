# Tensor Frame - Implementation TODO

## Critical Fixes

| Issue | Location | Description |
|-------|----------|-------------|
| Shape Validation | `src/tensor/shape.rs:9-11` | No validation for invalid shapes (e.g., zero dims) |
| Division by Zero | `src/backend/cpu.rs:92-109` | No check for division by zero |
| Axis Reductions | `src/backend/cpu.rs:111-134` | CPU backend missing axis-specific sum/mean |

## Essential Features

| Feature | Priority | Description |
|---------|----------|-------------|
| `matmul()` | HIGH | Matrix multiplication for 2D tensors |
| `bmm()` | HIGH | Batched matrix multiplication |
| Data Types | HIGH | Support beyond f32 (f64, i32, i64, etc.) |
| Indexing | HIGH | `tensor[i]`, `tensor[i:j]`, slice operations |
| Math Ops | HIGH | exp, log, pow, sqrt, sin, cos, relu, sigmoid |
| Random | MEDIUM | rand, randn, uniform initialization |
| Concat/Stack | MEDIUM | cat, stack, split, chunk operations |
| Conv2d | MEDIUM | Convolution operations for deep learning |
| Backend Selection | MEDIUM | `to_backend()`, `backend_type()` methods |

## Missing Operations

| Operation | Type | Example |
|-----------|------|---------|
| Comparisons | Element-wise | eq, ne, lt, gt, le, ge |
| Reductions | Aggregation | min, max, argmin, argmax, prod |
| Linear Algebra | Matrix ops | inverse, det, svd, eig |
| Statistical | Analysis | var, std, median, quantile |
| Logical | Boolean ops | logical_and, logical_or, logical_not |

## Performance

| Optimization | Impact | Description |
|--------------|--------|-------------|
| In-place ops | HIGH | Reduce memory allocation |
| Operation fusion | HIGH | Combine multiple ops into single kernel |
| BLAS integration | HIGH | Use optimized libraries for matmul |
| Memory pooling | MEDIUM | Reuse GPU memory allocations |
| Lazy evaluation | MEDIUM | Defer computation until needed |