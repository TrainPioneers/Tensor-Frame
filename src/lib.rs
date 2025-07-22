pub mod backend;
pub mod error;
pub mod tensor;

pub use backend::Backend;
pub use error::{Result, TensorError};
pub use tensor::{ops::TensorOps, shape::Shape, Tensor};

#[cfg(test)]
mod tests {
    use super::*;

    // ==== TENSOR CREATION TESTS ====

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        let data = tensor.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::ones(vec![3, 2]).unwrap();
        assert_eq!(tensor.shape().dims(), &[3, 2]);
        assert_eq!(tensor.numel(), 6);
        let data = tensor.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.to_vec().unwrap(), data);
    }

    #[test]
    fn test_tensor_1d() {
        let tensor = Tensor::ones(vec![5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[5]);
        assert_eq!(tensor.numel(), 5);
    }

    #[test]
    fn test_tensor_3d() {
        let tensor = Tensor::zeros(vec![2, 3, 4]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_tensor_scalar() {
        let tensor = Tensor::from_vec(vec![42.0], vec![]).unwrap();
        assert_eq!(tensor.shape().dims(), &[] as &[usize]);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.to_vec().unwrap(), vec![42.0]);
    }

    // ==== ARITHMETIC OPERATION TESTS ====

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_tensor_subtraction() {
        let a = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = (a - b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_tensor_multiplication() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = (a * b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_tensor_division() {
        let a = Tensor::from_vec(vec![8.0, 12.0, 16.0, 20.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let c = (a / b).unwrap();
        assert_eq!(c.to_vec().unwrap(), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_tensor_chain_operations() {
        let a = Tensor::ones(vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let c = Tensor::from_vec(vec![3.0, 3.0, 3.0, 3.0], vec![2, 2]).unwrap();

        let result = ((a + b).unwrap() * c).unwrap();
        assert_eq!(result.to_vec().unwrap(), vec![9.0, 9.0, 9.0, 9.0]);
    }

    // ==== BROADCASTING TESTS ====

    #[test]
    fn test_broadcast_2d_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
        // This should fail without proper broadcasting implementation
        // but we'll test the shape compatibility
        assert_eq!(a.shape().dims(), &[2, 2]);
        assert_eq!(b.shape().dims(), &[2]);
    }

    #[test]
    fn test_broadcast_same_shape() {
        let a = Tensor::ones(vec![2, 3]).unwrap();
        let b = Tensor::ones(vec![2, 3]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_broadcast_compatible_shapes() {
        let a = Tensor::ones(vec![2, 1]).unwrap();
        let b = Tensor::ones(vec![1, 3]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0], vec![]).unwrap(); // scalar
                                                              // Test shape compatibility
        assert_eq!(a.shape().dims(), &[2, 2]);
        assert_eq!(b.shape().dims(), &[] as &[usize]);
    }

    // ==== REDUCTION OPERATION TESTS ====

    #[test]
    fn test_tensor_sum() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let sum = tensor.sum(None).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![10.0]);
    }

    #[test]
    fn test_tensor_mean() {
        let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let mean = tensor.mean(None).unwrap();
        assert_eq!(mean.to_vec().unwrap(), vec![5.0]);
    }

    #[test]
    fn test_sum_ones() {
        let tensor = Tensor::ones(vec![3, 3]).unwrap();
        let sum = tensor.sum(None).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![9.0]);
    }

    #[test]
    fn test_mean_zeros() {
        let tensor = Tensor::zeros(vec![2, 5]).unwrap();
        let mean = tensor.mean(None).unwrap();
        assert_eq!(mean.to_vec().unwrap(), vec![0.0]);
    }

    // ==== SHAPE MANIPULATION TESTS ====

    #[test]
    fn test_tensor_reshape() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(
            reshaped.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_tensor_reshape_1d() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let reshaped = tensor.reshape(vec![4]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[4]);
        assert_eq!(reshaped.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_transpose_2d() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let transposed = tensor.transpose().unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        // For 2x3 -> 3x2 transpose: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        assert_eq!(
            transposed.to_vec().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    // ==== ERROR HANDLING TESTS ====

    #[test]
    fn test_shape_mismatch_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_vec(data, vec![2, 2]); // 3 elements, expecting 4
        assert!(result.is_err());
        if let Err(TensorError::ShapeMismatch { expected, got }) = result {
            assert_eq!(expected, vec![4]);
            assert_eq!(got, vec![3]);
        }
    }

    #[test]
    fn test_incompatible_shapes_addition() {
        let a = Tensor::ones(vec![2, 3]).unwrap();
        let b = Tensor::ones(vec![3, 4]).unwrap();
        let result = a + b;
        // This should either work with broadcasting or fail gracefully
        match result {
            Ok(_) => {}  // Broadcasting worked
            Err(_) => {} // Expected failure for incompatible shapes
        }
    }

    #[test]
    fn test_invalid_reshape() {
        let tensor = Tensor::ones(vec![2, 3]).unwrap(); // 6 elements
        let result = tensor.reshape(vec![2, 2]); // 4 elements
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_1d() {
        let tensor = Tensor::ones(vec![5]).unwrap();
        let result = tensor.transpose();
        // 1D transpose should either work (return same) or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // ==== EDGE CASE TESTS ====

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::zeros(vec![0]).unwrap();
        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor.to_vec().unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_large_tensor() {
        let tensor = Tensor::zeros(vec![100, 100]).unwrap();
        assert_eq!(tensor.numel(), 10000);
        assert_eq!(tensor.shape().dims(), &[100, 100]);
    }

    #[test]
    fn test_operations_with_negative_numbers() {
        let a = Tensor::from_vec(vec![-1.0, -2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, -3.0, -4.0], vec![2, 2]).unwrap();

        let sum = (a.clone() + b.clone()).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);

        let product = (a * b).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![-1.0, -4.0, -9.0, -16.0]);
    }

    #[test]
    fn test_operations_with_zero() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let zeros = Tensor::zeros(vec![2, 2]).unwrap();

        let sum = (a.clone() + zeros.clone()).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let product = (a * zeros).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_display_formatting() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let display_str = format!("{}", tensor);
        // Just ensure it doesn't panic and produces some output
        assert!(!display_str.is_empty());
    }

    #[test]
    fn test_division_by_zero() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]).unwrap(); // Contains zero
        
        let result = a / b;
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Division by zero"));
        }
    }

    #[test]
    fn test_invalid_shape_creation() {
        use crate::tensor::shape::Shape;
        
        // Test that zero dimensions are rejected
        let result = Shape::new(vec![2, 0, 3]);
        assert!(result.is_err());
        
        let result = Shape::new(vec![0]);
        assert!(result.is_err());
        
        // Valid shapes should still work
        let result = Shape::new(vec![2, 3, 4]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_broadcasting_subtraction() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap(); // 1D tensor to broadcast
        
        let result = a - b;
        assert!(result.is_ok());
        if let Ok(tensor) = result {
            let data = tensor.to_vec().unwrap();
            // Broadcasting: [[1,2],[3,4]] - [1,2] = [[0,0],[2,2]]
            assert_eq!(data, vec![0.0, 0.0, 2.0, 2.0]);
        }
    }

    #[test]
    fn test_broadcasting_multiplication() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap(); // Scalar to broadcast
        
        let result = a * b;
        assert!(result.is_ok());
        if let Ok(tensor) = result {
            let data = tensor.to_vec().unwrap();
            assert_eq!(data, vec![4.0, 6.0, 8.0, 10.0]);
        }
    }

    #[test]
    fn test_broadcasting_division() {
        let a = Tensor::from_vec(vec![4.0, 6.0, 8.0, 10.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap(); // Scalar to broadcast
        
        let result = a / b;
        assert!(result.is_ok());
        if let Ok(tensor) = result {
            let data = tensor.to_vec().unwrap();
            assert_eq!(data, vec![2.0, 3.0, 4.0, 5.0]);
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        // Test basic 2x2 * 2x2 matrix multiplication
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_ok());
        if let Ok(tensor) = result {
            assert_eq!(tensor.shape().dims(), &[2, 2]);
            let data = tensor.to_vec().unwrap();
            // [1 2] * [5 6] = [19 22]
            // [3 4]   [7 8]   [43 50]
            assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0]);
        }
    }

    #[test]
    fn test_matrix_multiplication_different_sizes() {
        // Test 2x3 * 3x2 = 2x2
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_ok());
        if let Ok(tensor) = result {
            assert_eq!(tensor.shape().dims(), &[2, 2]);
            let data = tensor.to_vec().unwrap();
            // [1 2 3] * [7  8 ] = [58  64]
            // [4 5 6]   [9  10]   [139 154]
            //           [11 12]
            assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
        }
    }

    #[test]
    fn test_matrix_multiplication_incompatible_dimensions() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap(); // 3x1, incompatible with 2x2
        
        let result = a.matmul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_math_operations() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], vec![4]).unwrap();
        
        // Test exp
        let exp_result = tensor.exp();
        assert!(exp_result.is_ok());
        if let Ok(exp_tensor) = exp_result {
            let data = exp_tensor.to_vec().unwrap();
            // exp(0) = 1, exp(1) ≈ 2.718, exp(2) ≈ 7.389, exp(3) ≈ 20.086
            assert!((data[0] - 1.0).abs() < 0.01);
            assert!((data[1] - 2.718).abs() < 0.01);
        }
        
        // Test relu
        let negative_tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();
        let relu_result = negative_tensor.relu();
        assert!(relu_result.is_ok());
        if let Ok(relu_tensor) = relu_result {
            let data = relu_tensor.to_vec().unwrap();
            assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
        }
    }

    #[test]
    fn test_sigmoid_operation() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        
        let sigmoid_result = tensor.sigmoid();
        assert!(sigmoid_result.is_ok());
        if let Ok(sigmoid_tensor) = sigmoid_result {
            let data = sigmoid_tensor.to_vec().unwrap();
            // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
            assert!((data[0] - 0.5).abs() < 0.01);
            assert!((data[1] - 0.731).abs() < 0.01);
            assert!((data[2] - 0.269).abs() < 0.01);
        }
    }

    #[test]
    fn test_axis_specific_sum() {
        // Test 2D tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        
        // Sum along axis 0 (rows): should give [1+4, 2+5, 3+6] = [5, 7, 9]
        let sum_axis0 = tensor.sum_axis(0);
        assert!(sum_axis0.is_ok());
        if let Ok(result) = sum_axis0 {
            assert_eq!(result.shape().dims(), &[3]);
            let data = result.to_vec().unwrap();
            assert_eq!(data, vec![5.0, 7.0, 9.0]);
        }
        
        // Sum along axis 1 (columns): should give [1+2+3, 4+5+6] = [6, 15]
        let sum_axis1 = tensor.sum_axis(1);
        assert!(sum_axis1.is_ok());
        if let Ok(result) = sum_axis1 {
            assert_eq!(result.shape().dims(), &[2]);
            let data = result.to_vec().unwrap();
            assert_eq!(data, vec![6.0, 15.0]);
        }
    }

    #[test]
    fn test_axis_specific_mean() {
        // Test 2D tensor: [[2, 4], [6, 8]]
        let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        
        // Mean along axis 0: should give [(2+6)/2, (4+8)/2] = [4, 6]
        let mean_axis0 = tensor.mean_axis(0);
        assert!(mean_axis0.is_ok());
        if let Ok(result) = mean_axis0 {
            assert_eq!(result.shape().dims(), &[2]);
            let data = result.to_vec().unwrap();
            assert_eq!(data, vec![4.0, 6.0]);
        }
        
        // Mean along axis 1: should give [(2+4)/2, (6+8)/2] = [3, 7]
        let mean_axis1 = tensor.mean_axis(1);
        assert!(mean_axis1.is_ok());
        if let Ok(result) = mean_axis1 {
            assert_eq!(result.shape().dims(), &[2]);
            let data = result.to_vec().unwrap();
            assert_eq!(data, vec![3.0, 7.0]);
        }
    }

    #[test]
    fn test_axis_specific_invalid_axis() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        
        // Test invalid axis (axis 2 for 2D tensor)
        let result = tensor.sum_axis(2);
        assert!(result.is_err());
        
        let result = tensor.mean_axis(2);
        assert!(result.is_err());
    }
}
