# WGPU Backend

The WGPU backend provides cross-platform GPU compute acceleration using the WebGPU standard. It supports Metal, Vulkan, DirectX 12, and OpenGL backends, making it an excellent choice for portable high-performance computing.

## Features

- **Cross-Platform**: Works on Windows, macOS, Linux, iOS, Android, and Web
- **Multiple APIs**: Supports Vulkan, Metal, DX12, DX11, OpenGL ES, and WebGL
- **Compute Shaders**: Uses WGSL (WebGPU Shading Language) for parallel operations
- **Memory Efficient**: GPU buffer management with automatic cleanup
- **Future-Proof**: Based on the emerging WebGPU standard

## Installation

Enable the WGPU backend with the feature flag:

```toml
[dependencies]
tensor_frame = { version = "0.0.1-alpha", features = ["wgpu"] }
```

**Additional Dependencies**:
- No platform-specific GPU drivers required
- Uses system graphics drivers (Metal, Vulkan, DirectX, OpenGL)

## System Requirements

### Minimum Requirements
- **GPU**: Any GPU with compute shader support
- **Driver**: Up-to-date graphics drivers
- **Memory**: Sufficient GPU memory for tensor data

### Supported Platforms

| Platform | Graphics API | Status |
|----------|-------------|--------|
| Windows | DirectX 12, Vulkan | ✅ Full support |
| Windows | DirectX 11 | ✅ Fallback support |  
| macOS | Metal | ✅ Full support |
| Linux | Vulkan | ✅ Full support |
| Linux | OpenGL ES | ⚠️ Limited support |
| iOS | Metal | ✅ Full support |
| Android | Vulkan, OpenGL ES | ✅ Full support |
| Web | WebGPU, WebGL2 | ⚠️ Experimental |

## Implementation Details

### Storage
WGPU tensors use GPU buffers for data storage:

```rust
pub struct WgpuStorage {
    pub buffer: Arc<wgpu::Buffer>,    // GPU buffer handle
}
```

**Buffer Properties**:
- **Location**: GPU memory (VRAM)
- **Layout**: Contiguous, row-major layout
- **Usage**: Storage buffers with compute shader access
- **Synchronization**: Automatic via command queue

### Compute Shaders
Operations are implemented as WGSL compute shaders loaded from external files in `src/shaders/`:

- `add.wgsl` - Element-wise addition
- `sub.wgsl` - Element-wise subtraction  
- `mul.wgsl` - Element-wise multiplication
- `div.wgsl` - Element-wise division with IEEE 754 compliance

```wgsl
// Example: Element-wise addition shader (add.wgsl)
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&input_a)) {
        return;
    }
    output[index] = input_a[index] + input_b[index];
}
```

### Parallelization
- **Workgroups**: Operations dispatched in parallel workgroups
- **Thread Count**: Automatically sized based on tensor dimensions
- **GPU Utilization**: Optimized for high occupancy on modern GPUs

## Performance Characteristics

### Strengths
- **Massive Parallelism**: Thousands of parallel threads
- **High Throughput**: Excellent for large tensor operations
- **Memory Bandwidth**: High GPU memory bandwidth utilization
- **Compute Density**: Specialized compute units for arithmetic operations

### Limitations
- **Latency**: GPU command submission and synchronization overhead
- **Memory Transfer**: CPU-GPU data transfers can be expensive
- **Limited Precision**: Currently only supports f32 operations
- **Shader Compilation**: First-use compilation overhead

### Performance Guidelines

#### Optimal Use Cases
```rust
// Large tensor operations (> 10K elements)
let large = Tensor::zeros(vec![2048, 2048])?;
let result = (large_a * large_b) + large_c;

// Repeated operations on same-sized tensors
for batch in batches {
    let output = model.forward(batch)?;  // Shader programs cached
}

// Element-wise operations with complex expressions
let result = ((a * b) + c).sqrt();  // Single GPU kernel
```

#### Suboptimal Use Cases
```rust
// Very small tensors
let small = Tensor::ones(vec![10, 10])?;  // GPU overhead dominates

// Frequent CPU-GPU transfers  
let gpu_tensor = cpu_tensor.to_backend(BackendType::Wgpu)?;
let back_to_cpu = gpu_tensor.to_vec()?;  // Expensive transfers

// Scalar operations
let sum = tensor.sum(None)?;  // Result copied back to CPU
```

## Memory Management

### GPU Memory Allocation
WGPU automatically manages GPU memory:

```rust
let tensor = Tensor::zeros(vec![2048, 2048])?;  // Allocates ~16MB GPU memory
```

**Memory Pool**: WGPU uses internal memory pools for efficient allocation  
**Garbage Collection**: Buffers automatically freed when last reference dropped  
**Fragmentation**: Large allocations may fail even with sufficient total memory

### Memory Transfer Patterns
```rust
// Efficient: Create on GPU
let gpu_tensor = Tensor::zeros(vec![1000, 1000])?
    .to_backend(BackendType::Wgpu)?;

// Inefficient: Frequent transfers
let result = cpu_data.to_backend(BackendType::Wgpu)?
    .sum(None)?
    .to_backend(BackendType::Cpu)?;  // Multiple transfers
```

### Memory Debugging
Monitor GPU memory usage:

```rust
// Check GPU memory limits
let limits = device.limits();
println!("Max buffer size: {} MB", limits.max_buffer_size / (1024*1024));

// Handle out-of-memory errors
match Tensor::zeros(vec![16384, 16384]) {
    Ok(tensor) => println!("Allocated 1GB GPU tensor"),
    Err(TensorError::BackendError(msg)) if msg.contains("memory") => {
        eprintln!("GPU out of memory, trying smaller size");
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Debugging and Profiling

### Shader Debugging
WGPU provides validation and debugging features:

```rust
// Enable validation (debug builds)
let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
    backends: wgpu::Backends::all(),
    flags: wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION,
    ..Default::default()
});
```

### Performance Profiling
Use GPU profiling tools:

**Windows (DirectX)**:
- PIX for Windows
- RenderDoc
- Visual Studio Graphics Diagnostics

**macOS (Metal)**:
- Xcode Instruments (GPU Timeline)
- Metal System Trace

**Linux (Vulkan)**:
- RenderDoc
- Vulkan Tools

### Custom Timing
```rust
use std::time::Instant;

let start = Instant::now();
let result = gpu_tensor_a + gpu_tensor_b;
// Note: GPU operations are asynchronous!
let _data = result.to_vec()?;  // Synchronization point
println!("GPU operation took: {:?}", start.elapsed());
```

## Error Handling

WGPU backend errors can occur at multiple levels:

### Device Creation Errors
```rust
match WgpuBackend::new() {
    Ok(backend) => println!("WGPU backend ready"),
    Err(TensorError::BackendError(msg)) => {
        eprintln!("WGPU initialization failed: {}", msg);
        // Fallback to CPU backend
    }
}
```

### Runtime Errors
```rust
// Out of GPU memory
let result = Tensor::zeros(vec![100000, 100000]); // May fail

// Shader compilation errors (rare)
let result = custom_operation(tensor);  // May fail for invalid shaders

// Device lost (driver reset, etc.)
let result = tensor.sum(None);  // May fail if device is lost
```

**Common Error Scenarios**:
- **Device Not Found**: No compatible GPU available
- **Out of Memory**: GPU memory exhausted
- **Driver Issues**: Outdated or buggy graphics drivers
- **Unsupported Operations**: Feature not implemented in WGPU backend

## Platform-Specific Notes

### Windows
- **DirectX 12**: Best performance and feature support
- **Vulkan**: Good alternative if DX12 not available
- **DirectX 11**: Fallback with limited compute support

### macOS
- **Metal**: Excellent native support and performance
- **MoltenVK**: Vulkan compatibility layer (not recommended for production)

### Linux
- **Vulkan**: Primary choice with best performance
- **OpenGL**: Fallback with limited compute features
- **Graphics Drivers**: Ensure latest Mesa/NVIDIA/AMD drivers

### Mobile (iOS/Android)
- **iOS**: Metal provides excellent mobile GPU performance
- **Android**: Vulkan on newer devices, OpenGL ES fallback
- **Power Management**: Be aware of thermal throttling

### Web (Experimental)
- **WebGPU**: Emerging standard with excellent performance potential
- **WebGL2**: Fallback with compute shader emulation
- **Browser Support**: Chrome/Edge (flag), Firefox (experimental)

## Optimization Tips

### Workgroup Size Tuning
```rust
// Optimal workgroup sizes depend on GPU architecture
// Current default: 64 threads per workgroup
// Nvidia: 32 (warp size) or 64
// AMD: 64 (wavefront size)  
// Intel: 32 or 64
// Mobile: 16 or 32
```

### Batch Operations
```rust
// Efficient: Batch similar operations
let results: Vec<Tensor> = inputs
    .iter()
    .map(|input| model.forward(input))
    .collect()?;

// Inefficient: Individual operations
for input in inputs {
    let result = model.forward(input)?;
    let cpu_result = result.to_vec()?;  // Forces synchronization
}
```

### Memory Layout Optimization
```rust
// Ensure tensor shapes are GPU-friendly
let aligned_size = (size + 63) & !63;  // Align to 64-element boundaries
let tensor = Tensor::zeros(vec![aligned_size, aligned_size])?;
```

## Future Developments

The WGPU backend is actively developed with planned improvements:

- **Reduction Operations**: Sum, mean, and other reductions on GPU
- **Advanced Operations**: GPU-optimized tensor operations
- **Mixed Precision**: f16 and bf16 data type support  
- **Async Operations**: Fully asynchronous GPU command queues
- **WebGPU Stability**: Production-ready web deployment