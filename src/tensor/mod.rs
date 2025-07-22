pub mod broadcast;
pub mod ops;
pub mod shape;

use crate::backend::{Storage, BACKENDS};
use crate::error::{Result, TensorError};
use broadcast::broadcast_data;
use ops::TensorOps;
use shape::Shape;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
}

impl Tensor {
    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        for backend in &BACKENDS[0..] {
            match backend.zeros(&shape) {
                Ok(storage) => return Ok(Tensor { storage, shape }),
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could create zeros tensor".to_string(),
        ))
    }

    pub fn ones(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        for backend in &BACKENDS[0..] {
            match backend.ones(&shape) {
                Ok(storage) => return Ok(Tensor { storage, shape }),
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
                Ok(storage) => return Ok(Tensor { storage, shape }),
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

    // Helper function for binary operations with broadcasting support
    fn binary_op_with_broadcast<F>(self, other: Self, op_name: &str, backend_op: F) -> Result<Self>
    where
        F: Fn(&dyn crate::backend::Backend, &Storage, &Storage) -> Result<Storage>,
    {
        // Try to find a compatible broadcast shape
        let result_shape = match self.shape.broadcast_shape(&other.shape) {
            Some(shape) => shape,
            None => {
                return Err(TensorError::BroadcastError(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    self.shape.dims(),
                    other.shape.dims()
                )));
            }
        };

        // If shapes are the same, use direct operation
        if self.shape == other.shape {
            for backend in &BACKENDS[0..] {
                match backend_op(backend.as_ref(), &self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self.shape,
                        })
                    }
                    Err(e) => {
                        // Propagate division by zero and similar critical errors immediately
                        if let TensorError::BackendError(msg) = &e {
                            if msg.contains("Division by zero") {
                                return Err(e);
                            }
                        }
                        continue;
                    }
                }
            }
        } else {
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
                        match backend_op(backend.as_ref(), &lhs_storage, &rhs_storage) {
                            Ok(storage) => {
                                return Ok(Tensor {
                                    storage,
                                    shape: result_shape,
                                })
                            }
                            Err(e) => {
                                // Propagate division by zero and similar critical errors immediately
                                if let TensorError::BackendError(msg) = &e {
                                    if msg.contains("Division by zero") {
                                        return Err(e);
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    _ => continue,
                }
            }
        }

        Err(TensorError::BackendError(format!(
            "No backend could perform {} operation", op_name
        )))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();

        // Check dimensions for matrix multiplication
        if self_dims.len() != 2 || other_dims.len() != 2 {
            return Err(TensorError::BackendError(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let m = self_dims[0]; // rows of A
        let k = self_dims[1]; // cols of A (must equal rows of B)
        let n = other_dims[1]; // cols of B

        if k != other_dims[0] {
            return Err(TensorError::DimensionMismatch {
                expected: k,
                got: other_dims[0],
            });
        }

        let result_shape = Shape::new(vec![m, n])?;

        for backend in &BACKENDS[0..] {
            match backend.matmul(&self.storage, &other.storage, &self.shape, &other.shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    })
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform matmul operation".to_string(),
        ))
    }

    // Math operations
    pub fn exp(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.exp(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform exp operation".to_string(),
        ))
    }

    pub fn log(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.log(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform log operation".to_string(),
        ))
    }

    pub fn sqrt(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sqrt(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sqrt operation".to_string(),
        ))
    }

    pub fn sin(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sin(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sin operation".to_string(),
        ))
    }

    pub fn cos(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.cos(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform cos operation".to_string(),
        ))
    }

    pub fn relu(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.relu(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform relu operation".to_string(),
        ))
    }

    pub fn sigmoid(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sigmoid(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    })
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sigmoid operation".to_string(),
        ))
    }
}

impl Add for Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: Self) -> Self::Output {
        self.binary_op_with_broadcast(other, "add", |backend, lhs, rhs| backend.add(lhs, rhs))
    }
}

impl Sub for Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: Self) -> Self::Output {
        self.binary_op_with_broadcast(other, "sub", |backend, lhs, rhs| backend.sub(lhs, rhs))
    }
}

impl Mul for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: Self) -> Self::Output {
        self.binary_op_with_broadcast(other, "mul", |backend, lhs, rhs| backend.mul(lhs, rhs))
    }
}

impl Div for Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: Self) -> Self::Output {
        self.binary_op_with_broadcast(other, "div", |backend, lhs, rhs| backend.div(lhs, rhs))
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
                    return Ok(Tensor { storage, shape });
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
                    return Ok(Tensor { storage, shape });
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

        write!(f, ", dtype=f32)")
    }
}
