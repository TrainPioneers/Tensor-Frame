use tensor_frame::{Tensor, TensorOps};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== New Tensor Operations Demo ===\n");

    // Matrix Multiplication
    println!("=== Matrix Multiplication ===");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

    println!("Matrix A (2x2):");
    println!("{}", a);
    println!("Matrix B (2x2):");
    println!("{}", b);

    let matmul_result = a.matmul(&b)?;
    println!("A @ B (matrix multiplication):");
    println!("{}\n", matmul_result);

    // Batched Matrix Multiplication
    println!("=== Batched Matrix Multiplication ===");
    let batch_a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], vec![2, 2, 2])?;
    let batch_b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2])?;

    println!("Batched matrices A (2 batches of 2x2):");
    println!("Batch 1: [[1, 2], [3, 4]]");
    println!("Batch 2: [[2, 3], [4, 5]]");
    println!("Batched matrices B (2 batches of 2x2):");
    println!("Batch 1: [[1, 0], [0, 1]] (identity)");
    println!("Batch 2: [[2, 0], [0, 2]] (scale by 2)");

    let bmm_result = batch_a.bmm(&batch_b)?;
    println!("Batched matrix multiplication result:");
    println!("Result shape: {:?}", bmm_result.shape().dims());
    println!("Data: {:?}\n", bmm_result.to_vec()?);

    // Mathematical Functions
    println!("=== Mathematical Functions ===");
    let math_tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4])?;
    println!("Original tensor: {}", math_tensor);

    let sqrt_result = math_tensor.sqrt()?;
    println!("Square root: {}", sqrt_result);

    let exp_result = math_tensor.exp()?;
    println!("Exponential: {}", exp_result);

    let log_result = math_tensor.log()?;
    println!("Natural log: {}", log_result);

    let pow_result = math_tensor.pow(0.5)?;
    println!("Power 0.5 (same as sqrt): {}\n", pow_result);

    // Trigonometric Functions
    println!("=== Trigonometric Functions ===");
    use std::f32::consts::PI;
    let trig_tensor = Tensor::from_vec(vec![0.0, PI / 4.0, PI / 2.0, PI], vec![4])?;
    println!("Angles (radians): {}", trig_tensor);

    let sin_result = trig_tensor.sin()?;
    println!("Sine: {}", sin_result);

    let cos_result = trig_tensor.cos()?;
    println!("Cosine: {}\n", cos_result);

    // Activation Functions
    println!("=== Activation Functions ===");
    let activation_tensor = Tensor::from_vec(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5])?;
    println!("Input: {}", activation_tensor);

    let relu_result = activation_tensor.relu()?;
    println!("ReLU: {}", relu_result);

    let sigmoid_result = activation_tensor.sigmoid()?;
    println!("Sigmoid: {}", sigmoid_result);

    let tanh_result = activation_tensor.tanh()?;
    println!("Tanh: {}\n", tanh_result);

    // Chained Operations
    println!("=== Chained Operations ===");
    let chain_tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0], vec![3])?;
    println!("Original: {}", chain_tensor);

    // Apply sqrt -> square (power 2) -> relu -> sigmoid
    let chain_result = chain_tensor.sqrt()?.pow(2.0)?.relu()?.sigmoid()?;
    println!("After sqrt -> pow(2) -> relu -> sigmoid: {}", chain_result);

    println!("\n=== Demo Complete ===");
    Ok(())
}
