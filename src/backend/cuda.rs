use super::{Backend, CudaStorage, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::{dtype::DType, shape::Shape};
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

        // Read the CUDA kernel source from the .cu file
        let kernel_source = include_str!("../kernels.cu");

        // Compile kernels using nvrtc
        let ptx = cudarc::nvrtc::compile_ptx(kernel_source).map_err(|e| {
            TensorError::BackendError(format!("Failed to compile CUDA kernels: {}", e))
        })?;

        // Load the module using the correct API
        let module = context
            .load_module(ptx)
            .map_err(|e| TensorError::BackendError(format!("Failed to load PTX module: {}", e)))?;

        let kernel_names = [
            "fill_ones_kernel",
            "add_kernel",
            "sub_kernel",
            "mul_kernel",
            "div_kernel",
            "sum_kernel",
            "transpose_2d_kernel",
        ];

        for &name in &kernel_names {
            let func = module.load_function(name).map_err(|e| {
                TensorError::BackendError(format!("Failed to get kernel {}: {}", name, e))
            })?;
            kernels.insert(name.to_string(), func);
        }

        println!("Successfully loaded {} CUDA kernels", kernels.len());
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

    fn zeros(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
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

    fn ones(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
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
                _ => Err(TensorError::BackendError(
                    "Failed to create CUDA storage".to_string(),
                )),
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
                _ => Err(TensorError::BackendError(
                    "Failed to create CUDA storage".to_string(),
                )),
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
                _ => Err(TensorError::BackendError(
                    "Failed to create CUDA storage".to_string(),
                )),
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

            // Create CUDA storage from the data
            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            // Now perform the operation
            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("div_kernel", a, b)
                }
                _ => Err(TensorError::BackendError(
                    "Failed to create CUDA storage".to_string(),
                )),
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

            match storage {
                Storage::Cuda(cuda_storage) => {
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
                _ => Err(TensorError::BackendError(
                    "CUDA backend can only operate on CUDA storage".to_string(),
                )),
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

            match storage {
                Storage::Cuda(cuda_storage) => {
                    let sum_result = self.sum(storage, axis)?;

                    if let Storage::Cuda(sum_storage) = sum_result {
                        let sum_data = self.to_vec_f32(&Storage::Cuda(sum_storage))?;
                        let mean_val = sum_data[0] / cuda_storage.buffer.len() as f32;

                        let stream = self.context.default_stream();
                        let result_buf = stream.memcpy_stod(&[mean_val]).map_err(|e| {
                            TensorError::BackendError(format!("Failed to copy mean to CUDA: {}", e))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    } else {
                        Err(TensorError::BackendError(
                            "Sum operation returned non-CUDA storage".to_string(),
                        ))
                    }
                }
                _ => Err(TensorError::BackendError(
                    "CUDA backend can only operate on CUDA storage".to_string(),
                )),
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
            match storage {
                Storage::Cuda(cuda_storage) => {
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
                        TensorError::BackendError(format!(
                            "Failed to launch transpose kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => Err(TensorError::BackendError(
                    "CUDA backend can only operate on CUDA storage".to_string(),
                )),
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
}
