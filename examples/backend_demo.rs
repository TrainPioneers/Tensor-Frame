use tensor_frame::{Tensor, TensorOps};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Backend Selection Demo ===\n");

    // Check available backends
    let available_backends = Tensor::available_backends();
    println!("Available backends: {:?}\n", available_backends);

    // Create a tensor
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    println!("Original tensor:");
    println!("Backend: {}", tensor.backend_type());
    println!("Data: {}\n", tensor);

    // Try to move to different backends
    for backend_name in &["CPU", "CUDA", "WGPU"] {
        println!("=== Attempting to move to {} backend ===", backend_name);
        
        match tensor.to_backend(backend_name) {
            Ok(moved_tensor) => {
                println!("✅ Successfully moved to {} backend", backend_name);
                println!("Backend type: {}", moved_tensor.backend_type());
                println!("Data preserved: {}", moved_tensor);
                
                // Test that operations work on the new backend
                let doubled = (moved_tensor.clone() * moved_tensor.clone())?;
                println!("Operations work: Element-wise square = {}", doubled);
                
                // Test matrix multiplication
                let identity = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])?;
                let identity_moved = identity.to_backend(backend_name).unwrap_or(identity);
                
                match moved_tensor.matmul(&identity_moved) {
                    Ok(matmul_result) => {
                        println!("Matrix multiplication works: A @ I = {}", matmul_result);
                    }
                    Err(e) => {
                        println!("Matrix multiplication failed: {}", e);
                    }
                }
                
                // Test mathematical functions
                let sqrt_result = moved_tensor.sqrt()?;
                println!("Math operations work: sqrt = {}", sqrt_result);
                
                println!();
            }
            Err(e) => {
                println!("❌ Failed to move to {} backend: {}\n", backend_name, e);
            }
        }
    }

    // Demonstrate cross-backend operations (fallback behavior)
    println!("=== Cross-Backend Operations ===");
    let tensor_a = Tensor::ones(vec![2, 2])?;
    let tensor_b = Tensor::ones(vec![2, 2])?;
    
    println!("Tensor A backend: {}", tensor_a.backend_type());
    println!("Tensor B backend: {}", tensor_b.backend_type());
    
    let result = (tensor_a + tensor_b)?;
    println!("Addition result backend: {}", result.backend_type());
    println!("Result: {}\n", result);

    // Performance comparison (if multiple backends available)
    if available_backends.len() > 1 {
        println!("=== Performance Comparison ===");
        let large_tensor = Tensor::ones(vec![100, 100])?;
        
        for backend_name in &available_backends {
            if let Ok(backend_tensor) = large_tensor.to_backend(backend_name) {
                let start = std::time::Instant::now();
                
                // Perform some operations
                let _result = backend_tensor
                    .pow(2.0)?
                    .sqrt()?
                    .relu()?;
                
                let duration = start.elapsed();
                println!("Backend {}: {:?}", backend_name, duration);
            }
        }
    }

    println!("\n=== Backend Selection Demo Complete ===");
    Ok(())
}