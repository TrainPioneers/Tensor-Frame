#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "wgpu")]
mod wgpu;

mod run_functions;

use std::fs;
use ::wgpu::{ShaderModuleDescriptor, ShaderSource};
use futures::executor::block_on;
use run_functions::*;
use crate::tensor::Tensor;
use crate::acceleration::wgpu::{setup_wgpu, run_on_wgpu};

fn run_operation (t1: Tensor, t2: Tensor, operation: RunOperation) -> Tensor {
    if cfg!(feature = "wgpu") {
        let setup = setup_wgpu();
        let shader_path = format!("instructions/wgpu/{:?}.wgsl", operation);
        let shader_code = fs::read_to_string(&shader_path)
            .expect(&format!("Failed to read shader file: {:?}", shader_path));
        let (device, queue) = block_on(setup);
        let shader_module =  device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Label"),
            source: ShaderSource::Wgsl(shader_code.into())
        });
        block_on(run_on_wgpu(t1,t2,device,queue,shader_module))
    } else if cfg!(feature = "cuda") {
        todo!();
        Tensor::zeros(Vec::new())
    } else {
        todo!();
        Tensor::zeros(Vec::new())
    }
}