// This module is used to run operations on the GPU.
use crate::tensor::{Tensor, TensorElement};
use cust::prelude::*;

pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

pub fn run<T>(a: &Tensor<T>, b: &Tensor<T>, operation: Operation) -> Tensor<T> where T: TensorElement {
    let a_device = to_device(a);
    let b_device = to_device(b);
    
    // Create output buffer on device
    let n = a.len();
    let mut result_device = DeviceBuffer::new(n).unwrap();
    
    // Configure kernel launch parameters
    let threads_per_block = 256;
    let blocks = (n as u32 + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    unsafe {
        let module = Module::load_from_file("instructions/cuda/kernel.ptx").unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        
        // Select kernel function based on operation
        let func_name = match operation {
            Operation::Add => "add",
            Operation::Sub => "sub",
            Operation::Mul => "mul",
            Operation::Div => "div",
        };
        
        let func = module.get_function(func_name).unwrap();
        
        func.launch(
            &stream,
            LaunchConfig {
                grid_size: (blocks, 1, 1),
                block_size: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                a_device.as_device_ptr(),
                b_device.as_device_ptr(),
                result_device.as_device_ptr(),
                n as i32,
            ),
        ).unwrap();
    }
    
    // Create new tensor from device buffer
    from_device(result_device)
}

// Then we can implement add using this run function
pub fn add<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: TensorElement {
    run(a, b, Operation::Add)
}

pub fn sub<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: TensorElement {
    run(a, b, Operation::Sub)
}

pub fn mul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: TensorElement {
    run(a, b, Operation::Mul)
}

pub fn div<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> where T: TensorElement {
    run(a, b, Operation::Div)
}

pub fn to_device<T: TensorElement>(tensor: &Tensor<T>) -> DeviceBuffer<T> {
    let data = tensor.data();
    let n = tensor.len();
    
    // Allocate device memory and copy data
    let mut device_buffer = DeviceBuffer::new(n).unwrap();
    device_buffer.copy_from(data).unwrap();
    
    device_buffer
}

pub fn from_device<T: TensorElement>(device_buffer: DeviceBuffer<T>) -> Tensor<T> {
    let n = device_buffer.len();
    let mut host_data = vec![T::default(); n];
    
    // Copy data back to host
    device_buffer.copy_to(&mut host_data).unwrap();
    
    Tensor::new(host_data)
}

