use std::error::Error;

use cust::memory::DeviceBuffer;

use util::ValidTensorType;

// Allocates device memory and copies data from host to device
pub fn allocate_and_copy_to_device<T>(data: &Vec<T>) -> Result<DeviceBuffer<T>, Box<dyn Error>>
where
    T: ValidTensorType,
{
    let mut device_buffer: DeviceBuffer<T> = DeviceBuffer::zeroed(data.len())?;
    device_buffer.copy_from_slice(&data)?;
    Ok(device_buffer)
}