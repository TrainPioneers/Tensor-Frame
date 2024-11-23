#[cfg ! (feature = "cpu")]
use cpu_vm::*;

#[cfg ! (feature = "cuda")]
use cuda_vm::*;

#[cfg ! (feature = "ocl")]
use ocl_vm::*;

#[cfg ! (feature = "hip")]
use hip_vm::*;

#[cfg ! (feature = "wgpu")]
use wgpu_vm::*;

use util::{RunOperation, ValidTensorType};


pub fn run_operation<T>(d1: Vec<T>, d2: Vec<T>, operation: RunOperation) -> Vec<T>
where
    T: ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64>,
{
    if cfg!(feature = "cpu") {
        todo!()
    } else if cfg!(feature = "cuda") {
        todo!()
    } else if cfg!(feature = "ocl") {
        todo!()
    } else if cfg!(feature = "hip") {
        todo!()
    } else if cfg!(feature = "wgpu") {
        todo!()
    } else {
        panic!("No feature found")
    }
}