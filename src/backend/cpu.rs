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

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| {
                if *y == 0.0 {
                    // Division by zero - return appropriate IEEE floating point value
                    if *x == 0.0 {
                        f32::NAN // 0/0 = NaN
                    } else if *x > 0.0 {
                        f32::INFINITY // positive/0 = +inf
                    } else {
                        f32::NEG_INFINITY // negative/0 = -inf
                    }
                } else {
                    x / y
                }
            })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn sum(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;

        match axis {
            None => {
                // Sum all elements
                let sum: f32 = data.iter().sum();
                Ok(Storage::Cpu(vec![sum]))
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

                Ok(Storage::Cpu(result))
            }
        }
    }

    fn mean(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        // Calculate sum first
        let sum_result = self.sum(storage, shape, axis)?;
        let sum_data = self.to_vec_f32(&sum_result)?;

        match axis {
            None => {
                // Mean of all elements
                let total_elements = shape.numel() as f32;
                let mean = sum_data[0] / total_elements;
                Ok(Storage::Cpu(vec![mean]))
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

                let axis_size = dims[axis_idx] as f32;
                let result: Vec<f32> = sum_data.iter().map(|&sum| sum / axis_size).collect();
                Ok(Storage::Cpu(result))
            }
        }
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
