use std::error::Error;

use cust::prelude::*;

use allocate_device::*;
use execute_kernel::*;
use execution_grid::*;
use load_kernel::*;
use util::ValidTensorType;

mod allocate_device;
mod load_kernel;
mod execution_grid;
mod execute_kernel;

// Main function to run the vector operation
fn run_vector_operation<T>(
    a: Vec<T>,
    b: Vec<T>,
    function_name: &str,
) -> Result<Vec<T>, Box<dyn Error>>
where
    T: ValidTensorType,
{
    if a.len() != b.len() {
        return Err("Vectors must have the same length".into());
    }

    // Allocate device memory and copy data to device
    let d_a = allocate_and_copy_to_device(&a)?;
    let d_b = allocate_and_copy_to_device(&b)?;
    let mut d_c: DeviceBuffer<T> = DeviceBuffer::zeroed(a.len())?;

    // Load the CUDA kernel
    let (module, kernel) = load_cuda_kernel(function_name)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Configure the execution grid
    let block_size = 256;
    let (grid_size, shared_memory_size) = configure_execution_grid::<T>(a.len(), block_size);

    // Execute the kernel
    execute_kernel(kernel, &d_a, &d_b, &mut d_c, &stream, grid_size, block_size, shared_memory_size)?;

    // Synchronize the stream to wait for completion
    stream.synchronize()?;

    // Copy the result back to host memory
    let mut c: Vec<T> = vec![T::default(); a.len()];
    d_c.copy_to(&mut c)?;

    Ok(c)
}
