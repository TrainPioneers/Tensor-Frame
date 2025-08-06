use super::{Backend, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::shape::Shape;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use {
    bytemuck, futures, tokio,
    wgpu::{Buffer, BufferUsages, Device, Queue},
};

#[derive(Debug)]
pub struct WgpuStorage {
    pub buffer: Arc<Buffer>,

    pub device: Arc<Device>,

    pub queue: Arc<Queue>,
    pub size: usize,
}

impl Clone for WgpuStorage {
    fn clone(&self) -> Self {
        WgpuStorage {
            buffer: self.buffer.clone(),

            device: self.device.clone(),

            queue: self.queue.clone(),
            size: self.size,
        }
    }
}

#[derive(Debug)]
pub struct WgpuBackend {
    device: Arc<Device>,

    queue: Arc<Queue>,
}

impl WgpuBackend {
    pub fn new_blocking() -> Result<Self> {
        {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                TensorError::BackendError(format!("Failed to create tokio runtime: {}", e))
            })?;
            rt.block_on(Self::new())
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| {
                TensorError::BackendError(format!("Failed to find suitable GPU adapter: {:?}", e))
            })?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Tensor Frame Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .map_err(|e| TensorError::BackendError(format!("Failed to create device: {}", e)))?;

        Ok(WgpuBackend {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    fn create_buffer(&self, data: &[f32]) -> Result<Buffer> {
        use wgpu::util::DeviceExt;

        // Handle empty buffers
        if data.is_empty() {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Empty Storage Buffer"),
                size: 4, // Minimum buffer size to avoid wgpu errors
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            return Ok(buffer);
        }

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

        Ok(buffer)
    }

    fn create_staging_buffer(&self, size: usize) -> Buffer {
        let buffer_size = if size == 0 {
            4 // Minimum buffer size
        } else {
            (size * std::mem::size_of::<f32>()) as u64
        };

        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn read_buffer(&self, buffer: &Buffer, size: usize) -> Result<Vec<f32>> {
        // Handle empty tensors
        if size == 0 {
            return Ok(Vec::new());
        }

        let staging_buffer = self.create_staging_buffer(size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Poll device in wgpu v25
        let _ = self.device.poll(wgpu::MaintainBase::Wait);

        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            TensorError::BackendError(format!("Failed to create tokio runtime: {}", e))
        })?;

        rt.block_on(receiver)
            .map_err(|e| {
                TensorError::BackendError(format!("Failed to receive mapping result: {}", e))
            })?
            .map_err(|e| TensorError::BackendError(format!("Failed to map buffer: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    fn create_compute_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry_point),
                    cache: None,
                    compilation_options: Default::default(),
                });

        Ok(compute_pipeline)
    }

    fn binary_operation(
        &self,
        lhs: &WgpuStorage,
        rhs: &WgpuStorage,
        operation: &str,
    ) -> Result<Storage> {
        if lhs.size != rhs.size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs.size],
                got: vec![rhs.size],
            });
        }

        // Read WGSL shader source from file based on operation
        let shader_filename = match operation {
            "add" => "add.wgsl",
            "sub" => "sub.wgsl",
            "mul" => "mul.wgsl",
            "div" => "div.wgsl",
            _ => {
                return Err(TensorError::BackendError(format!(
                    "Unknown operation: {}",
                    operation
                )))
            }
        };

        let mut shader_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        shader_path.push("src");
        shader_path.push("shaders");
        shader_path.push(shader_filename);

        let shader_source = fs::read_to_string(&shader_path).map_err(|e| {
            TensorError::BackendError(format!(
                "Failed to read shader file {}: {}",
                shader_path.display(),
                e
            ))
        })?;

        let pipeline = self.create_compute_pipeline(&shader_source, "main")?;

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: (lhs.size * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (lhs.size + 63) / 64; // Round up division
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(Storage::Wgpu(WgpuStorage {
            buffer: Arc::new(result_buffer),
            device: self.device.clone(),
            queue: self.queue.clone(),
            size: lhs.size,
        }))
    }
}

pub fn is_available() -> bool {
    {
        // Try to create a WGPU instance
        let instance = wgpu::Instance::default();
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(_) => return false,
        };

        rt.block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .is_ok()
        })
    }
    #[cfg(not(feature = "wgpu"))]
    false
}

impl Backend for WgpuBackend {
    fn is_available(&self) -> bool {
        true
    }

    fn zeros(&self, shape: &Shape) -> Result<Storage> {
        {
            let size = shape.numel();
            let data = vec![0.0f32; size];
            let buffer = self.create_buffer(&data)?;

            Ok(Storage::Wgpu(WgpuStorage {
                buffer: Arc::new(buffer),
                device: self.device.clone(),
                queue: self.queue.clone(),
                size,
            }))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn ones(&self, shape: &Shape) -> Result<Storage> {
        {
            let size = shape.numel();
            let data = vec![1.0f32; size];
            let buffer = self.create_buffer(&data)?;

            Ok(Storage::Wgpu(WgpuStorage {
                buffer: Arc::new(buffer),
                device: self.device.clone(),
                queue: self.queue.clone(),
                size,
            }))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
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

            let buffer = self.create_buffer(data)?;

            Ok(Storage::Wgpu(WgpuStorage {
                buffer: Arc::new(buffer),
                device: self.device.clone(),
                queue: self.queue.clone(),
                size: data.len(),
            }))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Convert storage to WGPU storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            // Create WGPU storages
            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage) else {
                unreachable!("WGPU backend should always create WGPU storage")
            };
            self.binary_operation(a, b, "add")
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage) else {
                unreachable!("WGPU backend should always create WGPU storage")
            };
            self.binary_operation(a, b, "sub")
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage) else {
                unreachable!("WGPU backend should always create WGPU storage")
            };
            self.binary_operation(a, b, "mul")
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;

            if lhs_data.len() != rhs_data.len() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![lhs_data.len()],
                    got: vec![rhs_data.len()],
                });
            }

            let shape = Shape::new(vec![lhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &shape)?;
            let rhs_storage = self.from_slice(&rhs_data, &shape)?;

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage) else {
                unreachable!("WGPU backend should always create WGPU storage")
            };
            self.binary_operation(a, b, "div")
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn sum(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            let data = self.to_vec_f32(storage)?;

            match axis {
                None => {
                    // Sum all elements
                    let sum: f32 = data.iter().sum();
                    let buffer = self.create_buffer(&[sum])?;

                    Ok(Storage::Wgpu(WgpuStorage {
                        buffer: Arc::new(buffer),
                        device: self.device.clone(),
                        queue: self.queue.clone(),
                        size: 1,
                    }))
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

                    let buffer = self.create_buffer(&result)?;
                    Ok(Storage::Wgpu(WgpuStorage {
                        buffer: Arc::new(buffer),
                        device: self.device.clone(),
                        queue: self.queue.clone(),
                        size: result_size,
                    }))
                }
            }
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn mean(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            match axis {
                None => {
                    // Mean of all elements
                    let data = self.to_vec_f32(storage)?;
                    let sum: f32 = data.iter().sum();
                    let mean = sum / data.len() as f32;
                    let buffer = self.create_buffer(&[mean])?;

                    Ok(Storage::Wgpu(WgpuStorage {
                        buffer: Arc::new(buffer),
                        device: self.device.clone(),
                        queue: self.queue.clone(),
                        size: 1,
                    }))
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

                    let buffer = self.create_buffer(&result)?;
                    Ok(Storage::Wgpu(WgpuStorage {
                        buffer: Arc::new(buffer),
                        device: self.device.clone(),
                        queue: self.queue.clone(),
                        size: result.len(),
                    }))
                }
            }
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage> {
        {
            let dims = shape.dims();
            if dims.len() != 2 {
                return Err(TensorError::BackendError(
                    "Transpose only supports 2D tensors".to_string(),
                ));
            }

            let rows = dims[0];
            let cols = dims[1];
            let data = self.to_vec_f32(storage)?;
            let mut result = vec![0.0f32; data.len()];

            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = data[i * cols + j];
                }
            }

            let buffer = self.create_buffer(&result)?;

            Ok(Storage::Wgpu(WgpuStorage {
                buffer: Arc::new(buffer),
                device: self.device.clone(),
                queue: self.queue.clone(),
                size: result.len(),
            }))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        match storage {
            Storage::Wgpu(wgpu_storage) => {
                self.read_buffer(&wgpu_storage.buffer, wgpu_storage.size)
            }
            #[cfg(feature = "cpu")]
            Storage::Cpu(data) => Ok(data.clone()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => Err(TensorError::BackendError(
                "Cannot convert CUDA storage with WGPU backend".to_string(),
            )),
        }
    }

    fn matmul(&self, _lhs: &Storage, _rhs: &Storage, _lhs_shape: &Shape, _rhs_shape: &Shape) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU matrix multiplication
            Err(TensorError::BackendError(
                "Matrix multiplication not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn bmm(&self, _lhs: &Storage, _rhs: &Storage, _lhs_shape: &Shape, _rhs_shape: &Shape) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU batched matrix multiplication
            Err(TensorError::BackendError(
                "Batched matrix multiplication not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn exp(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU exp function
            Err(TensorError::BackendError(
                "Exp function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn log(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU log function
            Err(TensorError::BackendError(
                "Log function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn sqrt(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU sqrt function
            Err(TensorError::BackendError(
                "Sqrt function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn pow(&self, _storage: &Storage, _power: f32) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU pow function
            Err(TensorError::BackendError(
                "Pow function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn sin(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU sin function
            Err(TensorError::BackendError(
                "Sin function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn cos(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU cos function
            Err(TensorError::BackendError(
                "Cos function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn relu(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU relu function
            Err(TensorError::BackendError(
                "ReLU function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn sigmoid(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU sigmoid function
            Err(TensorError::BackendError(
                "Sigmoid function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn tanh(&self, _storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "wgpu")]
        {
            // TODO: Implement WGPU tanh function
            Err(TensorError::BackendError(
                "Tanh function not yet implemented for WGPU backend".to_string(),
            ))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }
}
