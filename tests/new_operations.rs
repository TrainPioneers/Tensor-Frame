#[cfg(not(feature = "wgpu"))]
mod tests {
    use tensor_frame::{Tensor, TensorOps};
    #[test]
    fn test_matmul_basic() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        let expected = vec![19.0, 22.0, 43.0, 50.0]; // Manual calculation
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_matmul_different_shapes() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap(); // 2x3
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap(); // 3x2

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        // Expected: [[58, 64], [139, 154]]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_matmul_incompatible_shapes() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();

        let result = a.matmul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_non_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap(); // 1D

        let result = a.matmul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_bmm_basic() {
        // Create 2 batches of 2x2 matrices
        let a =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();
        let b =
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]).unwrap();

        let result = a.bmm(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2, 2]);

        // First batch: identity multiplication, second batch: scaling by 2
        let expected = vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0];
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_bmm_incompatible_batch_size() {
        let a = Tensor::ones(vec![2, 3, 4]).unwrap();
        let b = Tensor::ones(vec![3, 4, 5]).unwrap(); // Different batch size

        let result = a.bmm(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_exp_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
        let result = tensor.exp().unwrap();

        let actual = result.to_vec().unwrap();
        assert!((actual[0] - 1.0).abs() < 1e-6); // e^0 = 1
        assert!((actual[1] - std::f32::consts::E).abs() < 1e-6); // e^1 = e
        assert!((actual[2] - (std::f32::consts::E * std::f32::consts::E)).abs() < 1e-5);
        // e^2 = e^2
    }

    #[test]
    fn test_log_basic() {
        let tensor = Tensor::from_vec(
            vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)],
            vec![3],
        )
        .unwrap();
        let result = tensor.log().unwrap();

        let actual = result.to_vec().unwrap();
        assert!((actual[0] - 0.0).abs() < 1e-6); // ln(1) = 0
        assert!((actual[1] - 1.0).abs() < 1e-6); // ln(e) = 1
        assert!((actual[2] - 2.0).abs() < 1e-5); // ln(e^2) = 2
    }

    #[test]
    fn test_sqrt_basic() {
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4]).unwrap();
        let result = tensor.sqrt().unwrap();

        let expected = vec![1.0, 2.0, 3.0, 4.0];
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_pow_basic() {
        let tensor = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let result = tensor.pow(2.0).unwrap();

        let expected = vec![4.0, 9.0, 16.0];
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_sin_cos_basic() {
        use std::f32::consts::PI;

        let tensor = Tensor::from_vec(vec![0.0, PI / 2.0, PI], vec![3]).unwrap();

        let sin_result = tensor.sin().unwrap();
        let sin_actual = sin_result.to_vec().unwrap();
        assert!((sin_actual[0] - 0.0).abs() < 1e-6); // sin(0) = 0
        assert!((sin_actual[1] - 1.0).abs() < 1e-6); // sin(π/2) = 1
        assert!((sin_actual[2] - 0.0).abs() < 1e-6); // sin(π) = 0

        let cos_result = tensor.cos().unwrap();
        let cos_actual = cos_result.to_vec().unwrap();
        assert!((cos_actual[0] - 1.0).abs() < 1e-6); // cos(0) = 1
        assert!((cos_actual[1] - 0.0).abs() < 1e-6); // cos(π/2) = 0
        assert!((cos_actual[2] - (-1.0)).abs() < 1e-6); // cos(π) = -1
    }

    #[test]
    fn test_relu_basic() {
        let tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();
        let result = tensor.relu().unwrap();

        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_sigmoid_basic() {
        let tensor = Tensor::from_vec(vec![-10.0, 0.0, 10.0], vec![3]).unwrap();
        let result = tensor.sigmoid().unwrap();

        let actual = result.to_vec().unwrap();
        assert!(actual[0] < 0.001); // sigmoid(-10) ≈ 0
        assert!((actual[1] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
        assert!(actual[2] > 0.999); // sigmoid(10) ≈ 1
    }

    #[test]
    fn test_tanh_basic() {
        let tensor = Tensor::from_vec(vec![-10.0, 0.0, 10.0], vec![3]).unwrap();
        let result = tensor.tanh().unwrap();

        let actual = result.to_vec().unwrap();
        assert!(actual[0] < -0.999); // tanh(-10) ≈ -1
        assert!((actual[1] - 0.0).abs() < 1e-6); // tanh(0) = 0
        assert!(actual[2] > 0.999); // tanh(10) ≈ 1
    }

    #[test]
    fn test_chain_math_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0], vec![3]).unwrap();

        // Chain operations: sqrt -> exp -> log
        let result = tensor.sqrt().unwrap().exp().unwrap().log().unwrap();

        // Should return approximately the original sqrt values
        let expected = vec![1.0, 2.0, 3.0]; // sqrt of original values
        let actual = result.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", a, b);
        }
    }
}
