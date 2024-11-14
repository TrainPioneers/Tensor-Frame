use std::error::Error;

use cust::function::Function;
use cust::module::Module;

// Loads the CUDA kernel module
pub fn load_cuda_kernel(function_name: &str) -> Result<(Module, Function), Box<dyn Error>> {
    let cubin = std::fs::read("../../../instructions/cuda/kernel.cubin")?;
    let module = Module::from_cubin(&cubin, &[])?;
    let kernel = module.get_function(function_name)?;
    Ok((module, kernel))
}
