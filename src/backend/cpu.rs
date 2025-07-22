use super::{Backend, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::shape::Shape;

#[derive(Debug)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn zeros(&self, shape: &Shape) -> Result<Storage> {
        let size = shape.numel();
        Ok(Storage::Cpu(vec![0.0; size]))
    }

    fn ones(&self, shape: &Shape) -> Result<Storage> {
        let size = shape.numel();
        Ok(Storage::Cpu(vec![1.0; size]))
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        Ok(Storage::Cpu(data.to_vec()))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

        if lhs_data.len() != rhs_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs_data.len()],
                got: vec![rhs_data.len()],
            });
        }

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x + y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

        if lhs_data.len() != rhs_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs_data.len()],
                got: vec![rhs_data.len()],
            });
        }

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x - y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

        if lhs_data.len() != rhs_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs_data.len()],
                got: vec![rhs_data.len()],
            });
        }

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x * y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
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

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x / y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        
        if let Some(_axis_dim) = axis {
            // For now, return an error for axis-specific operations - need shape info
            return Err(TensorError::BackendError(
                "Axis-specific sum requires shape information - use tensor method instead".to_string(),
            ));
        }
        
        let sum: f32 = data.iter().sum();
        Ok(Storage::Cpu(vec![sum]))
    }

    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        
        if let Some(_axis_dim) = axis {
            // For now, return an error for axis-specific operations - need shape info
            return Err(TensorError::BackendError(
                "Axis-specific mean requires shape information - use tensor method instead".to_string(),
            ));
        }

        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        Ok(Storage::Cpu(vec![mean]))
    }

    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage> {
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(TensorError::BackendError(
                "Transpose only supports 2D tensors".to_string(),
            ));
        }

        let data = self.to_vec_f32(storage)?;
        let rows = dims[0];
        let cols = dims[1];
        let mut result = vec![0.0; data.len()];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = data[i * cols + j];
            }
        }

        Ok(Storage::Cpu(result))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        match storage {
            #[cfg(feature = "cpu")]
            Storage::Cpu(data) => Ok(data.clone()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => Err(TensorError::BackendError(
                "Cannot convert CUDA storage with CPU backend".to_string(),
            )),
            #[cfg(feature = "wgpu")]
            Storage::Wgpu(_) => Err(TensorError::BackendError(
                "Cannot convert WGPU storage with CPU backend".to_string(),
            )),
        }
    }

    fn matmul(&self, lhs: &Storage, rhs: &Storage, lhs_shape: &Shape, rhs_shape: &Shape) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

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

        // Perform matrix multiplication: C = A * B
        let mut result = vec![0.0; m * n];

        #[cfg(feature = "cpu")]
        {
            use rayon::prelude::*;
            
            // Use parallel iteration for better performance
            result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k_idx in 0..k {
                        sum += lhs_data[i * k + k_idx] * rhs_data[k_idx * n + j];
                    }
                    row[j] = sum;
                }
            });
        }

        #[cfg(not(feature = "cpu"))]
        {
            // Fallback without parallel processing
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k_idx in 0..k {
                        sum += lhs_data[i * k + k_idx] * rhs_data[k_idx * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }
        }

        Ok(Storage::Cpu(result))
    }

    // Math operations implementations
    fn exp(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|x| x.exp()).collect();
        Ok(Storage::Cpu(result))
    }

    fn log(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|x| x.ln()).collect();
        Ok(Storage::Cpu(result))
    }

    fn sqrt(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|x| x.sqrt()).collect();
        Ok(Storage::Cpu(result))
    }

    fn sin(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|x| x.sin()).collect();
        Ok(Storage::Cpu(result))
    }

    fn cos(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|x| x.cos()).collect();
        Ok(Storage::Cpu(result))
    }

    fn relu(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|x| x.max(0.0)).collect();
        Ok(Storage::Cpu(result))
    }

    fn sigmoid(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok(Storage::Cpu(result))
    }
}
