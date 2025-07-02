use tensor_frame::{Result, Tensor};

fn main() -> Result<()> {
    println!("🚀 Tensor Frame Broadcasting Examples");
    println!("=====================================\n");

    // Basic broadcasting: scalar with tensor
    println!("1. Scalar Broadcasting:");
    let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2])?;
    let scalar = Tensor::from_vec(vec![2.0], vec![])?;
    println!("   Tensor (2x2): {}", tensor);
    println!("   Scalar: {}", scalar);
    let result = (tensor.clone() / scalar.clone())?;
    println!("   Tensor / Scalar: {}\n", result);

    // Same shape operations
    println!("2. Same Shape Operations:");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    println!("   A: {}", a);
    println!("   B: {}", b);
    println!("   A + B: {}", (a.clone() + b.clone())?);
    println!("   A - B: {}", (a.clone() - b.clone())?);
    println!("   A * B: {}", (a.clone() * b.clone())?);
    println!("   B / A: {}\n", (b / a)?);

    // Broadcasting compatible shapes
    println!("3. Shape Broadcasting (2x1 with 1x3 -> 2x3):");
    let a = Tensor::from_vec(vec![10.0, 20.0], vec![2, 1])?;
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3])?;
    println!("   A (2x1): {}", a);
    println!("   B (1x3): {}", b);
    println!("   A + B: {}", (a.clone() + b.clone())?);
    println!("   A - B: {}", (a.clone() - b.clone())?);
    println!("   A * B: {}", (a.clone() * b.clone())?);
    println!("   A / B: {}\n", (a / b)?);

    // Broadcasting along single dimension
    println!("4. Broadcasting Along Single Dimension:");
    let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    let row = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3])?;
    println!("   Matrix (2x3): {}", matrix);
    println!("   Row vector (3): {}", row);
    let result = (matrix.clone() * row.clone())?;
    println!("   Matrix * Row (broadcasted): {}", result);

    // More complex broadcasting example
    println!("\n5. Complex Broadcasting Example:");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1])?;
    let b = Tensor::from_vec(vec![10.0, 20.0], vec![1, 2])?;
    println!("   A (3x1): {}", a);
    println!("   B (1x2): {}", b);
    let result = (a.clone() + b.clone())?;
    println!("   A + B (3x2): {}", result);

    // Test all operations
    println!("\n6. All Operations with Broadcasting:");
    let x = Tensor::from_vec(vec![100.0, 200.0], vec![2, 1])?;
    let y = Tensor::from_vec(vec![1.0, 2.0, 4.0], vec![1, 3])?;
    println!("   X (2x1): {}", x);
    println!("   Y (1x3): {}", y);
    println!("   X + Y: {}", (x.clone() + y.clone())?);
    println!("   X - Y: {}", (x.clone() - y.clone())?);
    println!("   X * Y: {}", (x.clone() * y.clone())?);
    println!("   X / Y: {}", (x / y)?);

    println!("\n✅ Broadcasting examples completed!");
    println!("Broadcasting is now supported for all arithmetic operations: +, -, *, /");

    Ok(())
}
