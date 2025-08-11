use tensor_frame::Tensor;

#[test]
fn test_backend_type() {
    let tensor = Tensor::ones(vec![2, 2]).unwrap();
    let backend_type = tensor.backend_type();

    // Should be one of the available backend types
    assert!(matches!(backend_type, "CPU" | "CUDA" | "WGPU"));
}

#[test]
fn test_available_backends() {
    let available = Tensor::available_backends();

    // Should have at least CPU backend (enabled by default)
    assert!(!available.is_empty());
    assert!(available.contains(&"CPU".to_string()));

    println!("Available backends: {:?}", available);
}

#[test]
fn test_to_backend_same_backend() {
    let tensor = Tensor::ones(vec![2, 2]).unwrap();
    let current_backend = tensor.backend_type();

    // Moving to the same backend should work
    let moved_tensor = tensor.to_backend(current_backend).unwrap();
    assert_eq!(moved_tensor.backend_type(), current_backend);
    assert_eq!(moved_tensor.to_vec().unwrap(), tensor.to_vec().unwrap());
}

#[test]
fn test_to_backend_cpu() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    // Should be able to move to CPU (always available)
    let cpu_tensor = tensor.to_backend("CPU").unwrap();
    assert_eq!(cpu_tensor.backend_type(), "CPU");
    assert_eq!(cpu_tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_to_backend_invalid() {
    let tensor = Tensor::ones(vec![2, 2]).unwrap();

    // Should fail for invalid backend name
    let result = tensor.to_backend("INVALID");
    assert!(result.is_err());
}

#[test]
fn test_backend_operations_preservation() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let original_data = tensor.to_vec().unwrap();
    let original_shape = tensor.shape().dims().to_vec();

    // Try to move to CPU backend
    if let Ok(cpu_tensor) = tensor.to_backend("CPU") {
        assert_eq!(cpu_tensor.to_vec().unwrap(), original_data);
        assert_eq!(cpu_tensor.shape().dims(), &original_shape);

        // Operations should still work
        let doubled = (cpu_tensor * tensor).unwrap();
        let expected: Vec<f32> = original_data.iter().map(|x| x * x).collect();
        let actual = doubled.to_vec().unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", a, b);
        }
    }
}
