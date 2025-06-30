pub mod broadcast;
pub mod dtype;
pub mod ops;
pub mod shape;

use crate::backend::{Storage, BACKENDS};
use crate::error::{Result, TensorError};
use broadcast::broadcast_data;
use dtype::DType;
use ops::TensorOps;
use shape::Shape;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    dtype: DType,
}

impl Tensor {
    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let dtype = DType::F32;
        for backend in &BACKENDS[0..] {
            match backend.zeros(&shape, dtype) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape,
                        dtype,
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could create zeros tensor".to_string(),
        ))
    }

    pub fn ones(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let dtype = DType::F32;
        for backend in &BACKENDS[0..] {
            match backend.ones(&shape, dtype) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape,
                        dtype,
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could create ones tensor".to_string(),
        ))
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        for backend in &BACKENDS[0..] {
            match backend.from_slice(&data, &shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape,
                        dtype: DType::F32,
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could create tensor from vector".to_string(),
        ))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        for backend in &BACKENDS[0..] {
            match backend.to_vec_f32(&self.storage) {
                Ok(vec) => return Ok(vec),
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could convert storage to Vec<f32>".to_string(),
        ))
    }
}

impl Add for Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: Self) -> Self::Output {
        // Check if shapes are compatible for broadcasting
        let result_shape = if self.shape == other.shape {
            self.shape.clone()
        } else if let Some(broadcasted_shape) = self.shape.broadcast_shape(&other.shape) {
            broadcasted_shape
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        };

        #[cfg(feature = "debug")]
        {
            println!(
                "Adding tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            );
            println!("Backend length: {}", BACKENDS.len());
        }

        // If shapes are the same, try backends directly
        if self.shape == other.shape {
            for backend in &BACKENDS[0..] {
                match backend.add(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self.shape,
                            dtype: self.dtype,
                        })
                    }
                    Err(_) => continue,
                }
            }
        }

        // Handle broadcasting by converting to CPU and using broadcast_data
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let (lhs_broadcasted, rhs_broadcasted) = broadcast_data(
            &self_data,
            &self.shape,
            &other_data,
            &other.shape,
            &result_shape,
        )?;

        // Create tensors with broadcasted data and try backends
        for backend in &BACKENDS[0..] {
            match (
                backend.from_slice(&lhs_broadcasted, &result_shape),
                backend.from_slice(&rhs_broadcasted, &result_shape),
            ) {
                (Ok(lhs_storage), Ok(rhs_storage)) => {
                    match backend.add(&lhs_storage, &rhs_storage) {
                        Ok(storage) => {
                            return Ok(Tensor {
                                storage,
                                shape: result_shape,
                                dtype: self.dtype,
                            })
                        }
                        Err(_) => continue,
                    }
                }
                _ => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform add operation".to_string(),
        ))
    }
}

impl Sub for Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        for backend in &BACKENDS[0..] {
            match backend.sub(&self.storage, &other.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape,
                        dtype: self.dtype,
                    })
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform sub operation".to_string(),
        ))
    }
}

impl Mul for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        for backend in &BACKENDS[0..] {
            match backend.mul(&self.storage, &other.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape,
                        dtype: self.dtype,
                    })
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform mul operation".to_string(),
        ))
    }
}

impl Div for Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        for backend in &BACKENDS[0..] {
            match backend.div(&self.storage, &other.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape,
                        dtype: self.dtype,
                    })
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform div operation".to_string(),
        ))
    }
}

impl TensorOps for Tensor {
    fn sum(&self, axis: Option<usize>) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sum(&self.storage, axis) {
                Ok(storage) => {
                    let shape = if axis.is_none() {
                        Shape::scalar()
                    } else {
                        self.shape.clone()
                    };
                    return Ok(Tensor {
                        storage,
                        shape,
                        dtype: self.dtype,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sum operation".to_string(),
        ))
    }

    fn mean(&self, axis: Option<usize>) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.mean(&self.storage, axis) {
                Ok(storage) => {
                    let shape = if axis.is_none() {
                        Shape::scalar()
                    } else {
                        self.shape.clone()
                    };
                    return Ok(Tensor {
                        storage,
                        shape,
                        dtype: self.dtype,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform mean operation".to_string(),
        ))
    }

    fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_shape = Shape::new(new_shape)?;
        if self.shape.numel() != new_shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.numel()],
                got: vec![new_shape.numel()],
            });
        }
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    fn transpose(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidShape(
                "Transpose only supports 2D tensors".to_string(),
            ));
        }
        for backend in &BACKENDS[0..] {
            match backend.transpose(&self.storage, &self.shape) {
                Ok(storage) => {
                    let dims = self.shape.dims();
                    let new_shape = Shape::new(vec![dims[1], dims[0]])?;
                    return Ok(Tensor {
                        storage,
                        shape: new_shape,
                        dtype: self.dtype,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform transpose operation".to_string(),
        ))
    }

    fn squeeze(&self, axis: Option<usize>) -> Result<Self> {
        let dims = self.shape.dims();
        let new_dims = if let Some(axis) = axis {
            if axis >= self.ndim() || dims[axis] != 1 {
                return Err(TensorError::InvalidShape(format!(
                    "Cannot squeeze axis {} with size {}",
                    axis, dims[axis]
                )));
            }
            dims.iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &d)| d)
                .collect()
        } else {
            dims.iter().filter(|&&d| d != 1).copied().collect()
        };

        let new_shape = Shape::new(new_dims)?;
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    fn unsqueeze(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(TensorError::InvalidShape(format!(
                "Axis {} out of range for {}D tensor",
                axis,
                self.ndim()
            )));
        }
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.insert(axis, 1);
        let new_shape = Shape::new(new_dims)?;
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
        })
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.to_vec().map_err(|_| fmt::Error)?;
        let shape = self.shape().dims();

        write!(f, "Tensor(")?;

        if shape.is_empty() {
            write!(f, "{:.4}", data[0])?;
        } else if shape.len() == 1 {
            write!(f, "[")?;
            for (i, &val) in data.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", val)?;
            }
            write!(f, "]")?;
        } else if shape.len() == 2 {
            write!(f, "[")?;
            for row in 0..shape[0] {
                if row > 0 {
                    write!(f, ",\n       ")?;
                }
                write!(f, "[")?;
                for col in 0..shape[1] {
                    if col > 0 {
                        write!(f, ", ")?;
                    }
                    let idx = row * shape[1] + col;
                    write!(f, "{:.4}", data[idx])?;
                }
                write!(f, "]")?;
            }
            write!(f, "]")?;
        } else {
            write!(f, "shape={:?}, data=[", shape)?;
            let max_display = 8.min(data.len());
            for (i, &val) in data.iter().take(max_display).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", val)?;
            }
            if data.len() > max_display {
                write!(f, ", ...")?;
            }
            write!(f, "]")?;
        }

        write!(f, ", dtype={})", self.dtype)
    }
}
