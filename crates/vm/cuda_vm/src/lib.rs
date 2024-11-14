use std::error::Error;

use cust::launch;
use cust::prelude::*;

use util::ValidTensorType;

fn run_vector_operation<T>(
    a: Vec<T>,
    b: Vec<T>,
    function_name: &str,
) -> Result<Vec<T>, Box<dyn Error>>
where
    T: ValidTensorType,
{
    // Number of elements in the vectors
    let n = a.len();

    // Ensure both vectors have the same size
    if b.len() != n {
        return Err("Vectors must have the same length".into());
    }

    // Allocate device memory (GPU)
    let mut d_a: DeviceBuffer<T> = DeviceBuffer::zeroed(n)?;
    let mut d_b: DeviceBuffer<T> = DeviceBuffer::zeroed(n)?;
    let mut d_c: DeviceBuffer<T> = DeviceBuffer::zeroed(n)?;

    // Copy data from host to device
    d_a.copy_from_slice(&a)?;
    d_b.copy_from_slice(&b)?;

    // Load the CUDA SASS kernel
    let cubin = std::fs::read("../../../instructions/cuda/kernel.cubin")?;
    let module = Module::from_cubin(&cubin, &[])?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get the kernel function
    let kernel = module.get_function(function_name)?;

    // Configure the execution grid
    let block_size = 256; // Number of threads per block
    //let grid_size = (n as T / block_size as T).ceil() as u32; // Number of blocks

    // Launch the kernel with the data
    unsafe {
        let result = launch!(kernel<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            d_c.len()
        ));
        result?;
    }

    stream.synchronize()?;


    // Allocate a host vector to store the result
    let mut c: Vec<T> = vec![0 as T; n];

    // Copy the result from device to host
    d_c.copy_to(&mut c)?;

    // Return the result
    Ok(c)
}

