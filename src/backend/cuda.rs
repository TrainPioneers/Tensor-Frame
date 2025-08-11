use super::{Backend, CudaStorage, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::shape::Shape;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};

use std::collections::HashMap;

#[derive(Debug)]
pub struct CudaBackend {
    context: std::sync::Arc<CudaContext>,

    kernels: HashMap<String, CudaFunction>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        {
            let context = CudaContext::new(0).map_err(|e| {
                TensorError::BackendError(format!("Failed to initialize CUDA: {}", e))
            })?;

            let kernels = Self::load_kernels(&context)?;
            Ok(CudaBackend { context, kernels })
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn load_kernels(
        context: &std::sync::Arc<CudaContext>,
    ) -> Result<HashMap<String, CudaFunction>> {
        let mut kernels = HashMap::new();

        // Define kernel files and their respective kernels
        let kernel_files = [
            (
                "fill",
                include_str!("../kernels/fill.cu"),
                vec!["fill_ones_kernel"],
            ),
            (
                "arithmetic",
                include_str!("../kernels/arithmetic.cu"),
                vec!["add_kernel", "sub_kernel", "mul_kernel", "div_kernel"],
            ),
            (
                "reduction",
                include_str!("../kernels/reduction.cu"),
                vec!["sum_kernel", "mean_kernel"],
            ),
            (
                "transform",
                include_str!("../kernels/transform.cu"),
                vec!["transpose_2d_kernel"],
            ),
            (
                "matmul",
                include_str!("../kernels/matmul.cu"),
                vec!["matmul_kernel", "matmul_tiled_kernel", "bmm_kernel"],
            ),
            (
                "math",
                include_str!("../kernels/math.cu"),
                vec![
                    "exp_kernel",
                    "log_kernel",
                    "sqrt_kernel",
                    "pow_kernel",
                    "sin_kernel",
                    "cos_kernel",
                    "relu_kernel",
                    "sigmoid_kernel",
                    "tanh_kernel",
                ],
            ),
        ];

        for (module_name, kernel_source, kernel_names) in &kernel_files {
            // Compile kernels using nvrtc
            let ptx = cudarc::nvrtc::compile_ptx(kernel_source).map_err(|e| {
                TensorError::BackendError(format!(
                    "Failed to compile CUDA kernels in {}: {}",
                    module_name, e
                ))
            })?;

            // Load the module using the correct API
            let module = context.load_module(ptx).map_err(|e| {
                TensorError::BackendError(format!(
                    "Failed to load PTX module {}: {}",
                    module_name, e
                ))
            })?;

            for &name in kernel_names {
                let func = module.load_function(name).map_err(|e| {
                    TensorError::BackendError(format!(
                        "Failed to get kernel {} from {}: {}",
                        name, module_name, e
                    ))
                })?;
                kernels.insert(name.to_string(), func);
            }
        }

        println!(
            "Successfully loaded {} CUDA kernels from {} modules",
            kernels.len(),
            kernel_files.len()
        );
        Ok(kernels)
    }

    fn launch_binary_kernel(
        &self,
        kernel_name: &str,
        a: &CudaStorage,
        b: &CudaStorage,
    ) -> Result<Storage> {
        if a.buffer.len() != b.buffer.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![a.buffer.len()],
                got: vec![b.buffer.len()],
            });
        }

        let stream = self.context.default_stream();
        let mut result_buf = stream.alloc_zeros::<f32>(a.buffer.len()).map_err(|e| {
            TensorError::BackendError(format!("Failed to allocate CUDA result buffer: {}", e))
        })?;

        let kernel = self.kernels.get(kernel_name).ok_or_else(|| {
            TensorError::BackendError(format!("Kernel {} not found", kernel_name))
        })?;

        let size = a.buffer.len();
        let cfg = LaunchConfig::for_num_elems(size as u32);

        let mut builder = stream.launch_builder(kernel);
        builder.arg(a.buffer.as_ref());
        builder.arg(b.buffer.as_ref());
        builder.arg(&mut result_buf);
        let size_arg = size as i32;
        builder.arg(&size_arg);

        unsafe { builder.launch(cfg) }.map_err(|e| {
            TensorError::BackendError(format!("Failed to launch {} kernel: {}", kernel_name, e))
        })?;

        Ok(Storage::Cuda(CudaStorage {
            buffer: std::sync::Arc::new(result_buf),
        }))
    }

    fn launch_unary_math_kernel(&self, kernel_name: &str, storage: &Storage) -> Result<Storage> {
        match storage {
            Storage::Cuda(cuda_storage) => {
                let stream = self.context.default_stream();
                let mut result_buf = stream
                    .alloc_zeros::<f32>(cuda_storage.buffer.len())
                    .map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                let kernel = self.kernels.get(kernel_name).ok_or_else(|| {
                    TensorError::BackendError(format!("{} not found", kernel_name))
                })?;

                let size = cuda_storage.buffer.len();
                let cfg = LaunchConfig::for_num_elems(size as u32);

                let mut builder = stream.launch_builder(kernel);
                builder.arg(cuda_storage.buffer.as_ref());
                builder.arg(&mut result_buf);
                let size_arg = size as i32;
                builder.arg(&size_arg);

                unsafe { builder.launch(cfg) }.map_err(|e| {
                    TensorError::BackendError(format!(
                        "Failed to launch {} kernel: {}",
                        kernel_name, e
                    ))
                })?;

                Ok(Storage::Cuda(CudaStorage {
                    buffer: std::sync::Arc::new(result_buf),
                }))
            }
            _ => {
                // Convert to CUDA and try again
                let data = self.to_vec_f32(storage)?;
                let shape = Shape::new(vec![data.len()])?;
                let cuda_storage = self.from_slice(&data, &shape)?;
                self.launch_unary_math_kernel(kernel_name, &cuda_storage)
            }
        }
    }
}

pub fn is_available() -> bool {
    {
        CudaContext::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

impl Backend for CudaBackend {
    fn is_available(&self) -> bool {
        is_available()
    }

    fn zeros(&self, shape: &Shape) -> Result<Storage> {
        {
            let size = shape.numel();
            let stream = self.context.default_stream();
            let buf = stream.alloc_zeros::<f32>(size).map_err(|e| {
                TensorError::BackendError(format!("Failed to allocate CUDA memory: {}", e))
            })?;
            Ok(Storage::Cuda(CudaStorage {
                buffer: std::sync::Arc::new(buf),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn ones(&self, shape: &Shape) -> Result<Storage> {
        {
            let size = shape.numel();
            let stream = self.context.default_stream();
            let mut buf = stream.alloc_zeros::<f32>(size).map_err(|e| {
                TensorError::BackendError(format!("Failed to allocate CUDA memory: {}", e))
            })?;

            let kernel = self.kernels.get("fill_ones_kernel").ok_or_else(|| {
                TensorError::BackendError("fill_ones_kernel not found".to_string())
            })?;

            let cfg = LaunchConfig::for_num_elems(size as u32);

            let mut builder = stream.launch_builder(kernel);
            builder.arg(&mut buf);
            let size_arg = size as i32;
            builder.arg(&size_arg);

            unsafe { builder.launch(cfg) }.map_err(|e| {
                TensorError::BackendError(format!("Failed to launch fill_ones kernel: {}", e))
            })?;

            Ok(Storage::Cuda(CudaStorage {
                buffer: std::sync::Arc::new(buf),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        {
            if data.len() != shape.numel() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![shape.numel()],
                    got: vec![data.len()],
                });
            }

            let stream = self.context.default_stream();
            let buf = stream.memcpy_stod(data).map_err(|e| {
                TensorError::BackendError(format!("Failed to copy data to CUDA: {}", e))
            })?;

            Ok(Storage::Cuda(CudaStorage {
                buffer: std::sync::Arc::new(buf),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Convert to vec, then to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            // Create CUDA storage from the data
            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            // Now perform the operation
            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("add_kernel", a, b)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Convert to vec, then to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            // Create CUDA storage from the data
            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            // Now perform the operation
            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("sub_kernel", a, b)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Convert to vec, then to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            // Create CUDA storage from the data
            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            // Now perform the operation
            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("mul_kernel", a, b)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Convert to vec, then to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            // Create CUDA storage from the data
            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            // Now perform the operation
            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("div_kernel", a, b)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sum(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match axis {
                None => {
                    // Sum all elements using CUDA kernel
                    let Storage::Cuda(cuda_storage) = storage;
                    {
                        let stream = self.context.default_stream();
                        let mut result_buf = stream.alloc_zeros::<f32>(1).map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                        let kernel = self.kernels.get("sum_kernel").ok_or_else(|| {
                            TensorError::BackendError("sum_kernel not found".to_string())
                        })?;

                        let size = cuda_storage.buffer.len();
                        let block_size = 256;
                        let grid_size = (size + block_size - 1) / block_size;

                        let cfg = LaunchConfig {
                            grid_dim: (grid_size as u32, 1, 1),
                            block_dim: (block_size as u32, 1, 1),
                            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
                        };

                        let mut builder = stream.launch_builder(kernel);
                        builder.arg(cuda_storage.buffer.as_ref());
                        builder.arg(&mut result_buf);
                        let size_arg = size as i32;
                        builder.arg(&size_arg);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            TensorError::BackendError(format!("Failed to launch sum kernel: {}", e))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
                }
                Some(axis_idx) => {
                    // Sum along specific axis
                    let dims = shape.dims();
                    if axis_idx >= dims.len() {
                        return Err(TensorError::InvalidShape(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            axis_idx,
                            dims.len()
                        )));
                    }

                    // Calculate result shape (remove the summed axis)
                    let mut result_shape = dims.to_vec();
                    result_shape.remove(axis_idx);
                    let result_size = if result_shape.is_empty() {
                        1
                    } else {
                        result_shape.iter().product()
                    };

                    // Convert CUDA storage to CPU, perform operation, then convert back
                    let data = self.to_vec_f32(storage)?;

                    // Calculate strides for the original tensor
                    let mut strides = vec![1; dims.len()];
                    for i in (0..dims.len() - 1).rev() {
                        strides[i] = strides[i + 1] * dims[i + 1];
                    }

                    let mut result = vec![0.0; result_size];

                    // Iterate through all elements and accumulate along the specified axis
                    for (linear_idx, &value) in data.iter().enumerate() {
                        // Convert linear index to multi-dimensional coordinates
                        let mut coords = vec![0; dims.len()];
                        let mut temp_idx = linear_idx;
                        for (i, &stride) in strides.iter().enumerate() {
                            coords[i] = temp_idx / stride;
                            temp_idx %= stride;
                        }

                        // Calculate result index by removing the summed axis coordinate
                        let mut result_coords = coords.clone();
                        result_coords.remove(axis_idx);

                        // Convert result coordinates to linear index
                        let mut result_idx = 0;
                        if !result_coords.is_empty() {
                            let mut result_strides = vec![1; result_coords.len()];
                            for i in (0..result_coords.len() - 1).rev() {
                                result_strides[i] = result_strides[i + 1] * result_shape[i + 1];
                            }
                            for (i, &coord) in result_coords.iter().enumerate() {
                                result_idx += coord * result_strides[i];
                            }
                        }

                        result[result_idx] += value;
                    }

                    // Convert result back to CUDA storage
                    let stream = self.context.default_stream();
                    let result_buf = stream.memcpy_stod(&result).map_err(|e| {
                        TensorError::BackendError(format!("Failed to copy result to CUDA: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn mean(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match axis {
                None => {
                    // Mean of all elements
                    let Storage::Cuda(cuda_storage) = storage;
                    {
                        let sum_result = self.sum(storage, shape, axis)?;

                        let Storage::Cuda(sum_storage) = sum_result;
                        let sum_data = self.to_vec_f32(&Storage::Cuda(sum_storage))?;
                        let mean_val = sum_data[0] / cuda_storage.buffer.len() as f32;

                        let stream = self.context.default_stream();
                        let result_buf = stream.memcpy_stod(&[mean_val]).map_err(|e| {
                            TensorError::BackendError(format!("Failed to copy mean to CUDA: {}", e))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
                }
                Some(axis_idx) => {
                    // Mean along specific axis
                    let dims = shape.dims();
                    if axis_idx >= dims.len() {
                        return Err(TensorError::InvalidShape(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            axis_idx,
                            dims.len()
                        )));
                    }

                    // First calculate sum, then divide by axis size
                    let sum_result = self.sum(storage, shape, Some(axis_idx))?;
                    let sum_data = self.to_vec_f32(&sum_result)?;
                    let axis_size = dims[axis_idx] as f32;
                    let result: Vec<f32> = sum_data.iter().map(|&sum| sum / axis_size).collect();

                    // Convert result back to CUDA storage
                    let stream = self.context.default_stream();
                    let result_buf = stream.memcpy_stod(&result).map_err(|e| {
                        TensorError::BackendError(format!("Failed to copy mean to CUDA: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage> {
        {
            let Storage::Cuda(cuda_storage) = storage;
            {
                let dims = shape.dims();
                if dims.len() != 2 {
                    return Err(TensorError::BackendError(
                        "Transpose only supports 2D tensors".to_string(),
                    ));
                }

                let rows = dims[0];
                let cols = dims[1];
                let stream = self.context.default_stream();
                let mut result_buf = stream
                    .alloc_zeros::<f32>(cuda_storage.buffer.len())
                    .map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                let kernel = self.kernels.get("transpose_2d_kernel").ok_or_else(|| {
                    TensorError::BackendError("transpose_2d_kernel not found".to_string())
                })?;

                let block_dim_x = 16;
                let block_dim_y = 16;
                let grid_dim_x = (cols + block_dim_x - 1) / block_dim_x;
                let grid_dim_y = (rows + block_dim_y - 1) / block_dim_y;

                let cfg = LaunchConfig {
                    grid_dim: (grid_dim_x as u32, grid_dim_y as u32, 1),
                    block_dim: (block_dim_x as u32, block_dim_y as u32, 1),
                    shared_mem_bytes: 0,
                };

                let mut builder = stream.launch_builder(kernel);
                builder.arg(cuda_storage.buffer.as_ref());
                builder.arg(&mut result_buf);
                let rows_arg = rows as i32;
                let cols_arg = cols as i32;
                builder.arg(&rows_arg);
                builder.arg(&cols_arg);

                unsafe { builder.launch(cfg) }.map_err(|e| {
                    TensorError::BackendError(format!("Failed to launch transpose kernel: {}", e))
                })?;

                Ok(Storage::Cuda(CudaStorage {
                    buffer: std::sync::Arc::new(result_buf),
                }))
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.context.default_stream();
                    let mut result = vec![0.0f32; cuda_storage.buffer.len()];
                    stream
                        .memcpy_dtoh(cuda_storage.buffer.as_ref(), &mut result)
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to copy data from CUDA device: {}",
                                e
                            ))
                        })?;
                    Ok(result)
                }
                #[cfg(feature = "cpu")]
                Storage::Cpu(data) => Ok(data.clone()),
                #[cfg(feature = "wgpu")]
                Storage::Wgpu(_) => Err(TensorError::BackendError(
                    "Cannot convert WGPU storage with CUDA backend".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn matmul(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            let lhs_dims = lhs_shape.dims();
            let rhs_dims = rhs_shape.dims();

            // Validate that tensors are 2D
            if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
                return Err(TensorError::InvalidShape(
                    "Matrix multiplication requires 2D tensors".to_string(),
                ));
            }

            // Validate dimensions for matrix multiplication: (M, K) x (K, N) -> (M, N)
            let (m, k1) = (lhs_dims[0], lhs_dims[1]);
            let (k2, n) = (rhs_dims[0], rhs_dims[1]);

            if k1 != k2 {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![k1],
                    got: vec![k2],
                });
            }

            // Convert inputs to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;
            let lhs_shape_single = Shape::new(vec![lhs_data.len()])?;
            let rhs_shape_single = Shape::new(vec![rhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &lhs_shape_single)?;
            let rhs_storage = self.from_slice(&rhs_data, &rhs_shape_single)?;

            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    let stream = self.context.default_stream();
                    let mut result_buf = stream.alloc_zeros::<f32>(m * n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    // Use tiled kernel for better performance
                    let kernel = self.kernels.get("matmul_tiled_kernel").ok_or_else(|| {
                        TensorError::BackendError("matmul_tiled_kernel not found".to_string())
                    })?;

                    let block_size = 16;
                    let grid_x = (n + block_size - 1) / block_size;
                    let grid_y = (m + block_size - 1) / block_size;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, 1),
                        block_dim: (block_size as u32, block_size as u32, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(a.buffer.as_ref());
                    builder.arg(b.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    let m_arg = m as i32;
                    let k_arg = k1 as i32;
                    let n_arg = n as i32;
                    builder.arg(&m_arg);
                    builder.arg(&k_arg);
                    builder.arg(&n_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!("Failed to launch matmul kernel: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => Err(TensorError::BackendError(
                    "Invalid storage types for CUDA matmul".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn bmm(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            let lhs_dims = lhs_shape.dims();
            let rhs_dims = rhs_shape.dims();

            // Validate that tensors are 3D
            if lhs_dims.len() != 3 || rhs_dims.len() != 3 {
                return Err(TensorError::InvalidShape(
                    "Batched matrix multiplication requires 3D tensors".to_string(),
                ));
            }

            // Validate dimensions: (B, M, K) x (B, K, N) -> (B, M, N)
            let (b1, m, k1) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
            let (b2, k2, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

            if b1 != b2 {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![b1],
                    got: vec![b2],
                });
            }

            if k1 != k2 {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![k1],
                    got: vec![k2],
                });
            }

            // Convert inputs to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;
            let lhs_shape_single = Shape::new(vec![lhs_data.len()])?;
            let rhs_shape_single = Shape::new(vec![rhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &lhs_shape_single)?;
            let rhs_storage = self.from_slice(&rhs_data, &rhs_shape_single)?;

            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    let stream = self.context.default_stream();
                    let mut result_buf = stream.alloc_zeros::<f32>(b1 * m * n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("bmm_kernel").ok_or_else(|| {
                        TensorError::BackendError("bmm_kernel not found".to_string())
                    })?;

                    let block_size = 16;
                    let grid_x = (n + block_size - 1) / block_size;
                    let grid_y = (m + block_size - 1) / block_size;
                    let grid_z = b1;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
                        block_dim: (block_size as u32, block_size as u32, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(a.buffer.as_ref());
                    builder.arg(b.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    let batch_arg = b1 as i32;
                    let m_arg = m as i32;
                    let k_arg = k1 as i32;
                    let n_arg = n as i32;
                    builder.arg(&batch_arg);
                    builder.arg(&m_arg);
                    builder.arg(&k_arg);
                    builder.arg(&n_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!("Failed to launch bmm kernel: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => Err(TensorError::BackendError(
                    "Invalid storage types for CUDA bmm".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn exp(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("exp_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn log(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("log_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sqrt(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("sqrt_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn pow(&self, storage: &Storage, power: f32) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.context.default_stream();
                    let mut result_buf = stream
                        .alloc_zeros::<f32>(cuda_storage.buffer.len())
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                    let kernel = self.kernels.get("pow_kernel").ok_or_else(|| {
                        TensorError::BackendError("pow_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let cfg = LaunchConfig::for_num_elems(size as u32);

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&power);
                    let size_arg = size as i32;
                    builder.arg(&size_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!("Failed to launch pow kernel: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, power)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sin(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("sin_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn cos(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("cos_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn relu(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("relu_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sigmoid(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("sigmoid_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn tanh(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("tanh_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }
}
