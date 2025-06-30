use super::{Backend, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::{dtype::DType, shape::Shape};

#[derive(Debug)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn zeros(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
        let size = shape.numel();
        Ok(Storage::Cpu(vec![0.0; size]))
    }

    fn ones(&self, shape: &Shape, _dtype: DType) -> Result<Storage> {
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

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x / y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        if axis.is_some() {
            return Err(TensorError::BackendError(
                "Axis sum not yet implemented".to_string(),
            ));
        }

        let data = self.to_vec_f32(storage)?;
        let sum: f32 = data.iter().sum();
        Ok(Storage::Cpu(vec![sum]))
    }

    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage> {
        if axis.is_some() {
            return Err(TensorError::BackendError(
                "Axis mean not yet implemented".to_string(),
            ));
        }

        let data = self.to_vec_f32(storage)?;
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
}
