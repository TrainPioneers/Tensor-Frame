//! Tensor operations trait defining common tensor operations.
//! 
//! This module provides the [`TensorOps`] trait which defines the interface
//! for various tensor operations including reductions, shape manipulations,
//! and transformations.

use crate::error::Result;

/// Trait defining common operations on tensors.
/// 
/// This trait provides a standard interface for tensor operations that can be
/// implemented by different tensor types. All operations return a `Result` to
/// handle potential errors gracefully.
/// 
/// # Examples
/// 
/// ```
/// use tensor_frame::{Tensor, TensorOps};
/// 
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// 
/// // Sum all elements
/// let sum = tensor.sum(None).unwrap();
/// assert_eq!(sum.to_vec().unwrap(), vec![10.0]);
/// 
/// // Reshape the tensor
/// let reshaped = tensor.reshape(vec![4]).unwrap();
/// assert_eq!(reshaped.shape().dims(), &[4]);
/// ```
pub trait TensorOps {
    /// Computes the sum of tensor elements.
    /// 
    /// # Arguments
    /// 
    /// * `axis` - Optional axis along which to sum. If `None`, sums all elements.
    /// 
    /// # Returns
    /// 
    /// A tensor containing the sum. If summing all elements, returns a scalar tensor.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// 
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let sum = tensor.sum(None).unwrap();
    /// assert_eq!(sum.to_vec().unwrap(), vec![10.0]);
    /// ```
    fn sum(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;
    
    /// Computes the mean of tensor elements.
    /// 
    /// # Arguments
    /// 
    /// * `axis` - Optional axis along which to compute mean. If `None`, computes mean of all elements.
    /// 
    /// # Returns
    /// 
    /// A tensor containing the mean. If computing mean of all elements, returns a scalar tensor.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// 
    /// let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
    /// let mean = tensor.mean(None).unwrap();
    /// assert_eq!(mean.to_vec().unwrap(), vec![5.0]);
    /// ```
    fn mean(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;
    
    /// Reshapes the tensor to a new shape.
    /// 
    /// The new shape must have the same total number of elements as the original.
    /// 
    /// # Arguments
    /// 
    /// * `new_shape` - The desired shape
    /// 
    /// # Returns
    /// 
    /// A tensor with the new shape containing the same data.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the new shape has a different number of elements.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// 
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let reshaped = tensor.reshape(vec![3, 2]).unwrap();
    /// assert_eq!(reshaped.shape().dims(), &[3, 2]);
    /// ```
    fn reshape(&self, new_shape: Vec<usize>) -> Result<Self>
    where
        Self: Sized;
    
    /// Transposes the tensor.
    /// 
    /// Currently only supports 2D tensors. For a 2D tensor, swaps rows and columns.
    /// 
    /// # Returns
    /// 
    /// A new tensor with transposed dimensions.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the tensor is not 2D.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// 
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let transposed = tensor.transpose().unwrap();
    /// assert_eq!(transposed.shape().dims(), &[3, 2]);
    /// ```
    fn transpose(&self) -> Result<Self>
    where
        Self: Sized;
    
    /// Removes dimensions of size 1 from the tensor shape.
    /// 
    /// # Arguments
    /// 
    /// * `axis` - Optional specific axis to squeeze. If `None`, removes all dimensions of size 1.
    /// 
    /// # Returns
    /// 
    /// A tensor with squeezed dimensions.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the specified axis doesn't have size 1.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// 
    /// let tensor = Tensor::ones(vec![2, 1, 3]).unwrap();
    /// let squeezed = tensor.squeeze(Some(1)).unwrap();
    /// assert_eq!(squeezed.shape().dims(), &[2, 3]);
    /// ```
    fn squeeze(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;
    
    /// Adds a dimension of size 1 at the specified position.
    /// 
    /// # Arguments
    /// 
    /// * `axis` - The position where to insert the new dimension
    /// 
    /// # Returns
    /// 
    /// A tensor with an additional dimension of size 1.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the axis is out of range.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// 
    /// let tensor = Tensor::ones(vec![2, 3]).unwrap();
    /// let unsqueezed = tensor.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed.shape().dims(), &[2, 1, 3]);
    /// ```
    fn unsqueeze(&self, axis: usize) -> Result<Self>
    where
        Self: Sized;
}
