use util::ValidTensorType;

// Configures grid and block sizes for kernel execution
pub fn configure_execution_grid<T>(n: usize, block_size: u32) -> (u32, usize)
where
    T: ValidTensorType,
{
    let grid_size = ((n as f64 / block_size as f64).ceil()) as u32;
    let shared_memory_size = block_size as usize * std::mem::size_of::<T>();
    (grid_size, shared_memory_size)
}