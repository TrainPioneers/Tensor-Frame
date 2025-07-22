use super::{Backend, CudaStorage, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::shape::Shape;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[derive(Debug)]
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    context: std::sync::Arc<CudaContext>,
    #[cfg(feature = "cuda")]
    kernels: HashMap<String, CudaFunction>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
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
                vec!["matmul_kernel", "matmul_shared_kernel"],
            ),
            (
                "math",
                include_str!("../kernels/math.cu"),
                vec!["exp_kernel", "log_kernel", "sqrt_kernel", "sin_kernel", "cos_kernel", "relu_kernel", "sigmoid_kernel"],
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

    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
    fn launch_unary_kernel(
        &self,
        kernel_name: &str,
        input: &CudaStorage,
    ) -> Result<Storage> {
        let stream = self.context.default_stream();
        let mut result_buf = stream.alloc_zeros::<f32>(input.buffer.len()).map_err(|e| {
            TensorError::BackendError(format!("Failed to allocate CUDA result buffer: {}", e))
        })?;

        let kernel = self.kernels.get(kernel_name).ok_or_else(|| {
            TensorError::BackendError(format!("Kernel {} not found", kernel_name))
        })?;

        let size = input.buffer.len();
        let cfg = LaunchConfig::for_num_elems(size as u32);

        let mut builder = stream.launch_builder(kernel);
        builder.arg(input.buffer.as_ref());
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
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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

            // Check for division by zero
            if rhs_data.iter().any(|&y| y == 0.0) {
                return Err(TensorError::BackendError(
                    "Division by zero detected".to_string(),
                ));
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

    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            if axis.is_some() {
                return Err(TensorError::BackendError(
                    "Axis sum not yet implemented for CUDA".to_string(),
                ));
            }

            let Storage::Cuda(cuda_storage) = storage;
            {
                let stream = self.context.default_stream();
                let mut result_buf = stream.alloc_zeros::<f32>(1).map_err(|e| {
                    TensorError::BackendError(format!(
                        "Failed to allocate CUDA result buffer: {}",
                        e
                    ))
                })?;

                let kernel = self
                    .kernels
                    .get("sum_kernel")
                    .ok_or_else(|| TensorError::BackendError("sum_kernel not found".to_string()))?;

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
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            if axis.is_some() {
                return Err(TensorError::BackendError(
                    "Axis mean not yet implemented for CUDA".to_string(),
                ));
            }

            let Storage::Cuda(cuda_storage) = storage;
            {
                let sum_result = self.sum(storage, axis)?;

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
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage> {
        #[cfg(feature = "cuda")]
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
        #[cfg(feature = "cuda")]
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

    fn matmul(&self, lhs: &Storage, rhs: &Storage, lhs_shape: &Shape, rhs_shape: &Shape) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            let lhs_dims = lhs_shape.dims();
            let rhs_dims = rhs_shape.dims();

            // Check dimensions for matrix multiplication
            if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
                return Err(TensorError::BackendError(
                    "Matrix multiplication requires 2D tensors".to_string(),
                ));
            }

            let m = lhs_dims[0]; // rows of A
            let k = lhs_dims[1]; // cols of A (must equal rows of B)
            let n = rhs_dims[1]; // cols of B

            if k != rhs_dims[0] {
                return Err(TensorError::DimensionMismatch {
                    expected: k,
                    got: rhs_dims[0],
                });
            }

            // Ensure we have CUDA storage for both operands
            let lhs_storage = match lhs {
                Storage::Cuda(_) => lhs.clone(),
                _ => {
                    let lhs_data = self.to_vec_f32(lhs)?;
                    self.from_slice(&lhs_data, lhs_shape)?
                }
            };

            let rhs_storage = match rhs {
                Storage::Cuda(_) => rhs.clone(),
                _ => {
                    let rhs_data = self.to_vec_f32(rhs)?;
                    self.from_slice(&rhs_data, rhs_shape)?
                }
            };

            // Now extract CUDA buffers
            let (lhs_cuda, rhs_cuda) = match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(lhs_cuda), Storage::Cuda(rhs_cuda)) => (lhs_cuda, rhs_cuda),
                _ => return Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
            };

            let stream = self.context.default_stream();
            let mut result_buf = stream.alloc_zeros::<f32>(m * n).map_err(|e| {
                TensorError::BackendError(format!("Failed to allocate CUDA result buffer: {}", e))
            })?;

            // Choose kernel based on matrix size
            let kernel_name = if m >= 64 && n >= 64 && k >= 64 {
                "matmul_shared_kernel" // Use shared memory for larger matrices
            } else {
                "matmul_kernel" // Use simple kernel for smaller matrices
            };

            let kernel = self.kernels.get(kernel_name).ok_or_else(|| {
                TensorError::BackendError(format!("Kernel {} not found", kernel_name))
            })?;

            // Configure launch parameters
            let block_size = 16; // Using 16x16 blocks
            let grid_x = (n + block_size - 1) / block_size;
            let grid_y = (m + block_size - 1) / block_size;

            let cfg = LaunchConfig {
                grid_dim: (grid_x as u32, grid_y as u32, 1),
                block_dim: (block_size as u32, block_size as u32, 1),
                shared_mem_bytes: if kernel_name == "matmul_shared_kernel" {
                    2 * block_size * block_size * std::mem::size_of::<f32>()
                } else {
                    0
                } as u32,
            };

            let mut builder = stream.launch_builder(kernel);
            builder.arg(lhs_cuda.buffer.as_ref());
            builder.arg(rhs_cuda.buffer.as_ref());
            builder.arg(&mut result_buf);
            let m_arg = m as i32;
            let n_arg = n as i32;
            let k_arg = k as i32;
            builder.arg(&m_arg);
            builder.arg(&n_arg);
            builder.arg(&k_arg);

            unsafe { builder.launch(cfg) }.map_err(|e| {
                TensorError::BackendError(format!("Failed to launch {} kernel: {}", kernel_name, e))
            })?;

            Ok(Storage::Cuda(CudaStorage {
                buffer: std::sync::Arc::new(result_buf),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    // Math operations implementations
    fn exp(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    self.launch_unary_kernel("exp_kernel", cuda_storage)
                }
                _ => {
                    // Convert to CUDA storage first
                    let data = self.to_vec_f32(storage)?;
                    let cuda_storage = self.from_slice(&data, &Shape::new(vec![data.len()])?)?;
                    match cuda_storage {
                        Storage::Cuda(cuda_storage) => self.launch_unary_kernel("exp_kernel", &cuda_storage),
                        _ => Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn log(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    self.launch_unary_kernel("log_kernel", cuda_storage)
                }
                _ => {
                    let data = self.to_vec_f32(storage)?;
                    let cuda_storage = self.from_slice(&data, &Shape::new(vec![data.len()])?)?;
                    match cuda_storage {
                        Storage::Cuda(cuda_storage) => self.launch_unary_kernel("log_kernel", &cuda_storage),
                        _ => Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sqrt(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    self.launch_unary_kernel("sqrt_kernel", cuda_storage)
                }
                _ => {
                    let data = self.to_vec_f32(storage)?;
                    let cuda_storage = self.from_slice(&data, &Shape::new(vec![data.len()])?)?;
                    match cuda_storage {
                        Storage::Cuda(cuda_storage) => self.launch_unary_kernel("sqrt_kernel", &cuda_storage),
                        _ => Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
                    }
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
            match storage {
                Storage::Cuda(cuda_storage) => {
                    self.launch_unary_kernel("sin_kernel", cuda_storage)
                }
                _ => {
                    let data = self.to_vec_f32(storage)?;
                    let cuda_storage = self.from_slice(&data, &Shape::new(vec![data.len()])?)?;
                    match cuda_storage {
                        Storage::Cuda(cuda_storage) => self.launch_unary_kernel("sin_kernel", &cuda_storage),
                        _ => Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn cos(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    self.launch_unary_kernel("cos_kernel", cuda_storage)
                }
                _ => {
                    let data = self.to_vec_f32(storage)?;
                    let cuda_storage = self.from_slice(&data, &Shape::new(vec![data.len()])?)?;
                    match cuda_storage {
                        Storage::Cuda(cuda_storage) => self.launch_unary_kernel("cos_kernel", &cuda_storage),
                        _ => Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn relu(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    self.launch_unary_kernel("relu_kernel", cuda_storage)
                }
                _ => {
                    let data = self.to_vec_f32(storage)?;
                    let cuda_storage = self.from_slice(&data, &Shape::new(vec![data.len()])?)?;
                    match cuda_storage {
                        Storage::Cuda(cuda_storage) => self.launch_unary_kernel("relu_kernel", &cuda_storage),
                        _ => Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sigmoid(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    self.launch_unary_kernel("sigmoid_kernel", cuda_storage)
                }
                _ => {
                    let data = self.to_vec_f32(storage)?;
                    let cuda_storage = self.from_slice(&data, &Shape::new(vec![data.len()])?)?;
                    match cuda_storage {
                        Storage::Cuda(cuda_storage) => self.launch_unary_kernel("sigmoid_kernel", &cuda_storage),
                        _ => Err(TensorError::BackendError("Failed to create CUDA storage".to_string())),
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }
}
