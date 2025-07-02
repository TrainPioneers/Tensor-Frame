use super::{Backend, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::shape::Shape;
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

        let shader_source = format!(
            r#"
            @group(0) @binding(0) var<storage, read> input_a: array<f32>;
            @group(0) @binding(1) var<storage, read> input_b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let index = global_id.x;
                if (index >= arrayLength(&output)) {{
                    return;
                }}
                
                output[index] = input_a[index] {} input_b[index];
            }}
        "#,
            match operation {
                "add" => "+",
                "sub" => "-",
                "mul" => "*",
                "div" => "/",
                _ =>
                    return Err(TensorError::BackendError(format!(
                        "Unknown operation: {}",
                        operation
                    ))),
            }
        );

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

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage);
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

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage);
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

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage);
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

            let (Storage::Wgpu(a), Storage::Wgpu(b)) = (&lhs_storage, &rhs_storage);
            self.binary_operation(a, b, "div")
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        
        {
            if axis.is_some() {
                return Err(TensorError::BackendError(
                    "Axis sum not yet implemented".to_string(),
                ));
            }

            let data = self.to_vec_f32(storage)?;
            let sum: f32 = data.iter().sum();
            let buffer = self.create_buffer(&[sum])?;

            Ok(Storage::Wgpu(WgpuStorage {
                buffer: Arc::new(buffer),
                device: self.device.clone(),
                queue: self.queue.clone(),
                size: 1,
            }))
        }
        #[cfg(not(feature = "wgpu"))]
        Err(TensorError::BackendError(
            "WGPU support not compiled in".to_string(),
        ))
    }

    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        
        {
            if axis.is_some() {
                return Err(TensorError::BackendError(
                    "Axis mean not yet implemented".to_string(),
                ));
            }

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
}
