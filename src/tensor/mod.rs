//! The main tensor module containing the core tensor type and its implementations.
//!
//! This module provides the [`Tensor`] struct which is the central data structure
//! in the library, along with its associated operations and traits.

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

/// A multi-dimensional array with support for various backends and operations.
///
/// The `Tensor` struct is the core data structure of this library. It consists of:
/// - `storage`: Backend-specific storage for the tensor data
/// - `shape`: The dimensions of the tensor
///
/// Tensors support various operations including element-wise arithmetic, broadcasting,
/// reductions, and shape manipulation. The actual computation is delegated to the
/// available backends (CPU, WGPU, or CUDA).
///
/// # Examples
///
/// ```
/// use tensor_frame::Tensor;
///
/// // Create a 2x3 tensor of ones
/// let tensor = Tensor::ones(vec![2, 3]).unwrap();
///
/// // Create from data
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
}

impl Tensor {
    /// Creates a new tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result` containing the new tensor or an error if no backend could create it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let zeros = Tensor::zeros(vec![3, 4]).unwrap();
    /// assert_eq!(zeros.shape().dims(), &[3, 4]);
    /// ```
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

    /// Creates a new tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result` containing the new tensor or an error if no backend could create it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let ones = Tensor::ones(vec![2, 2]).unwrap();
    /// assert_eq!(ones.to_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
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

    /// Creates a new tensor from a vector of data with the specified shape.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to fill the tensor with
    /// * `shape` - The shape of the tensor
    ///
    /// # Returns
    ///
    /// A `Result` containing the new tensor or an error if:
    /// - The data length doesn't match the shape
    /// - No backend could create the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();
    /// assert_eq!(tensor.shape().dims(), &[2, 3]);
    /// ```
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

    /// Returns a reference to the tensor's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the number of dimensions of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 3, 4]).unwrap();
    /// assert_eq!(tensor.ndim(), 3);
    /// ```
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 3, 4]).unwrap();
    /// assert_eq!(tensor.numel(), 24);  // 2 * 3 * 4
    /// ```
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Creates a new tensor with explicit shape validation.
    ///
    /// This method validates the shape before creating the tensor, allowing for
    /// better error handling than the `From` trait implementations.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to fill the tensor with
    /// * `dims` - The dimensions of the tensor (will be validated)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(tensor.shape().dims(), &[2, 2]);
    ///
    /// // This will fail with proper error handling
    /// let result = Tensor::from_vec_with_shape(vec![1.0, 2.0], vec![0, 2]);
    /// assert!(result.is_err());
    /// ```
    pub fn from_vec_with_shape(data: Vec<f32>, dims: Vec<usize>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        Self::from_vec(data, shape)
    }

    /// Converts the tensor to a vector of f32 values.
    ///
    /// The data is returned in row-major (C-style) order.
    ///
    /// # Returns
    ///
    /// A `Result` containing the data as a vector or an error if no backend
    /// could perform the conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let data = tensor.to_vec().unwrap();
    /// assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
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

    /// Gets the backend type currently used by this tensor.
    ///
    /// # Returns
    ///
    /// A string identifying the backend type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 2]).unwrap();
    /// let backend_type = tensor.backend_type();
    /// // Returns "CPU", "CUDA", or "WGPU" depending on available backends
    /// ```
    pub fn backend_type(&self) -> &'static str {
        match &self.storage {
            #[cfg(feature = "cpu")]
            Storage::Cpu(_) => "CPU",
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => "CUDA",
            #[cfg(feature = "wgpu")]
            Storage::Wgpu(_) => "WGPU",
        }
    }

    /// Attempts to move this tensor to a specific backend.
    ///
    /// This method tries to convert the tensor to use the specified backend.
    /// If the backend is not available or the conversion fails, returns an error.
    ///
    /// # Arguments
    ///
    /// * `backend_name` - The name of the target backend ("CPU", "CUDA", or "WGPU")
    ///
    /// # Returns
    ///
    /// A new tensor using the specified backend, or an error if conversion fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 2]).unwrap();
    /// // Try to move to CUDA backend (if available)
    /// match tensor.to_backend("CUDA") {
    ///     Ok(cuda_tensor) => println!("Successfully moved to CUDA"),
    ///     Err(e) => println!("Could not move to CUDA: {}", e),
    /// }
    /// ```
    pub fn to_backend(&self, backend_name: &str) -> Result<Self> {
        // If already on the requested backend, return a clone
        if self.backend_type() == backend_name {
            return Ok(self.clone());
        }

        // Get the data from current backend
        let data = self.to_vec()?;

        // Find the requested backend and create tensor with it
        for backend in &BACKENDS[0..] {
            match backend_name {
                "CPU" => {
                    #[cfg(feature = "cpu")]
                    if let Ok(storage) = backend.from_slice(&data, &self.shape) {
                        if matches!(storage, Storage::Cpu(_)) {
                            return Ok(Tensor {
                                storage,
                                shape: self.shape.clone(),
                            });
                        }
                    }
                    continue;
                }
                "CUDA" => {
                    #[cfg(feature = "cuda")]
                    if let Ok(storage) = backend.from_slice(&data, &self.shape) {
                        if matches!(storage, Storage::Cuda(_)) {
                            return Ok(Tensor {
                                storage,
                                shape: self.shape.clone(),
                            });
                        }
                    }
                    continue;
                }
                "WGPU" => {
                    #[cfg(feature = "wgpu")]
                    if let Ok(storage) = backend.from_slice(&data, &self.shape) {
                        if matches!(storage, Storage::Wgpu(_)) {
                            return Ok(Tensor {
                                storage,
                                shape: self.shape.clone(),
                            });
                        }
                    }
                    continue;
                }
                _ => {
                    return Err(TensorError::BackendError(format!(
                        "Unknown backend: {}. Supported backends: CPU, CUDA, WGPU",
                        backend_name
                    )));
                }
            }
        }

        Err(TensorError::BackendError(format!(
            "Backend {} is not available or failed to create tensor",
            backend_name
        )))
    }

    /// Lists all available backends on the current system.
    ///
    /// # Returns
    ///
    /// A vector of backend names that are available and working.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let available = Tensor::available_backends();
    /// println!("Available backends: {:?}", available);
    /// ```
    pub fn available_backends() -> Vec<String> {
        let mut available = Vec::new();
        
        for backend in &BACKENDS[0..] {
            if backend.is_available() {
                // Test create a small tensor to verify the backend works
                if let Ok(storage) = backend.zeros(&Shape::new(vec![1]).unwrap()) {
                    match storage {
                        #[cfg(feature = "cpu")]
                        Storage::Cpu(_) => available.push("CPU".to_string()),
                        #[cfg(feature = "cuda")]
                        Storage::Cuda(_) => available.push("CUDA".to_string()),
                        #[cfg(feature = "wgpu")]
                        Storage::Wgpu(_) => available.push("WGPU".to_string()),
                    }
                }
            }
        }
        
        available
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

        // If shapes are the same, try backends directly
        if self.shape == other.shape {
            for backend in &BACKENDS[0..] {
                match backend.sub(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self.shape,
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
                    match backend.sub(&lhs_storage, &rhs_storage) {
                        Ok(storage) => {
                            return Ok(Tensor {
                                storage,
                                shape: result_shape,
                            })
                        }
                        Err(_) => continue,
                    }
                }
                _ => continue,
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

        // If shapes are the same, try backends directly
        if self.shape == other.shape {
            for backend in &BACKENDS[0..] {
                match backend.mul(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self.shape,
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
                    match backend.mul(&lhs_storage, &rhs_storage) {
                        Ok(storage) => {
                            return Ok(Tensor {
                                storage,
                                shape: result_shape,
                            })
                        }
                        Err(_) => continue,
                    }
                }
                _ => continue,
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

        // If shapes are the same, try backends directly
        if self.shape == other.shape {
            for backend in &BACKENDS[0..] {
                match backend.div(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self.shape,
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
                    match backend.div(&lhs_storage, &rhs_storage) {
                        Ok(storage) => {
                            return Ok(Tensor {
                                storage,
                                shape: result_shape,
                            })
                        }
                        Err(_) => continue,
                    }
                }
                _ => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform div operation".to_string(),
        ))
    }
}

impl TensorOps for Tensor {
    fn sum(&self, axis: Option<usize>) -> Result<Self> {
        // Calculate the result shape
        let result_shape = match axis {
            None => Shape::scalar(),
            Some(axis_idx) => {
                let dims = self.shape.dims();
                if axis_idx >= dims.len() {
                    return Err(TensorError::InvalidShape(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis_idx,
                        dims.len()
                    )));
                }
                // Remove the summed axis from the shape
                let mut result_dims = dims.to_vec();
                result_dims.remove(axis_idx);
                Shape::new(result_dims)?
            }
        };

        for backend in &BACKENDS[0..] {
            match backend.sum(&self.storage, &self.shape, axis) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
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
        // Calculate the result shape (same logic as sum)
        let result_shape = match axis {
            None => Shape::scalar(),
            Some(axis_idx) => {
                let dims = self.shape.dims();
                if axis_idx >= dims.len() {
                    return Err(TensorError::InvalidShape(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis_idx,
                        dims.len()
                    )));
                }
                // Remove the averaged axis from the shape
                let mut result_dims = dims.to_vec();
                result_dims.remove(axis_idx);
                Shape::new(result_dims)?
            }
        };

        for backend in &BACKENDS[0..] {
            match backend.mean(&self.storage, &self.shape, axis) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
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

    fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(TensorError::InvalidShape(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();
        
        // Validate dimensions: (M, K) × (K, N) → (M, N)
        if self_dims[1] != other_dims[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_dims[1]],
                got: vec![other_dims[0]],
            });
        }

        let result_shape = Shape::new(vec![self_dims[0], other_dims[1]])?;

        for backend in &BACKENDS[0..] {
            match backend.matmul(&self.storage, &other.storage, &self.shape, &other.shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform matrix multiplication".to_string(),
        ))
    }

    fn bmm(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 3 || other.ndim() != 3 {
            return Err(TensorError::InvalidShape(
                "Batched matrix multiplication requires 3D tensors".to_string(),
            ));
        }

        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();
        
        // Validate dimensions: (B, M, K) × (B, K, N) → (B, M, N)
        if self_dims[0] != other_dims[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_dims[0]],
                got: vec![other_dims[0]],
            });
        }
        
        if self_dims[2] != other_dims[1] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_dims[2]],
                got: vec![other_dims[1]],
            });
        }

        let result_shape = Shape::new(vec![self_dims[0], self_dims[1], other_dims[2]])?;

        for backend in &BACKENDS[0..] {
            match backend.bmm(&self.storage, &other.storage, &self.shape, &other.shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform batched matrix multiplication".to_string(),
        ))
    }

    fn exp(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.exp(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform exp operation".to_string(),
        ))
    }

    fn log(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.log(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform log operation".to_string(),
        ))
    }

    fn sqrt(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sqrt(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sqrt operation".to_string(),
        ))
    }

    fn pow(&self, power: f32) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.pow(&self.storage, power) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform pow operation".to_string(),
        ))
    }

    fn sin(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sin(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sin operation".to_string(),
        ))
    }

    fn cos(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.cos(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform cos operation".to_string(),
        ))
    }

    fn relu(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.relu(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform relu operation".to_string(),
        ))
    }

    fn sigmoid(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sigmoid(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sigmoid operation".to_string(),
        ))
    }

    fn tanh(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.tanh(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform tanh operation".to_string(),
        ))
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
