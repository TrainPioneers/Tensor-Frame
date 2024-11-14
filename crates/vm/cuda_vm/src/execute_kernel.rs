use std::error::Error;

use cust::function::Function;
use cust::launch;
use cust::memory::DeviceBuffer;
use cust::prelude::Stream;

use util::ValidTensorType;

// Executes the kernel on the GPU
pub fn execute_kernel<T>(
    kernel: Function,
    d_a: &DeviceBuffer<T>,
    d_b: &DeviceBuffer<T>,
    d_c: &mut DeviceBuffer<T>,
    stream: &Stream,
    grid_size: u32,
    block_size: u32,
    shared_memory_size: usize,
) -> Result<(), Box<dyn Error>>
where
    T: ValidTensorType,
{
    unsafe {
        launch!(kernel<<<grid_size, block_size, shared_memory_size, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            d_c.len()
        ))?;
    }
    Ok(())
}
