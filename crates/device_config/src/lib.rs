#[cfg(feature = "cpu")]
use cpu_vm::run_vector_operation;

#[cfg(feature = "cuda")]
use cuda_vm::run_vector_operation;

#[cfg(feature = "ocl")]
use ocl_vm::run_vector_operation;

#[cfg(feature = "hip")]
use hip_vm::run_vector_operation;

#[cfg(feature = "wgpu")]
use wgpu_vm::run_vector_operation;

use util::{RunOperation, ValidTensorType};


pub fn run_operation<T>(d1: Vec<T>, d2: Vec<T>, operation: RunOperation) -> Vec<T>
where
    T: ValidTensorType + Clone + Default,
{
    #[cfg(feature = "cpu")]
    println!("feature cpu!!!!");

    #[cfg(feature = "cuda")]
    println!("feature cuda!!!!");

    #[cfg(feature = "ocl")]
    println!("feature ocl!!!!");

    #[cfg(feature = "hip")]
    println!("feature hip!!!!");

    #[cfg(feature = "wgpu")]
    println!("feature wgpu!!!!");
    if cfg!(feature = "cpu") {
        let binding = format!("{:?}", operation).to_lowercase();
        let op = binding.as_str();
        return run_vector_operation(d1, d2, op).unwrap();
    } else if cfg!(feature = "cuda") {
        todo!()
    } else if cfg!(feature = "ocl") {
        todo!()
    } else if cfg!(feature = "hip") {
        todo!()
    } else if cfg!(feature = "wgpu") {
        todo!()
    } else {
        println!("{},{},{},{},{}", cfg!(feature = "cpu"), cfg!(feature = "cuda"), cfg!(feature = "ocl"), cfg!(feature = "hip"), cfg!(feature = "wgpu"));
        panic!("No feature found")
    }
}

// Fallback method
#[cfg(not(
    any(feature = "cpu", feature = "cuda", feature = "ocl", feature = "hip", feature = "wgpu")
))]
pub fn run_vector_operation<T>(_d1: Vec<T>, _d2: Vec<T>, _op: &str) -> Result<Vec<T>, &'static str> {
    Err("No valid backend available for vector operation")
}
