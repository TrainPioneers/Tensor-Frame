use std::collections::HashMap;
use std::fs;
use once_cell::sync::Lazy;
use crate::acceleration::run_functions::RunOperation;

pub(crate) static SHADER_CACHE: Lazy<HashMap<RunOperation, String>> = Lazy::new(|| {
    let mut cache = HashMap::new();

    // Preload shaders
    for op in [RunOperation::Add, RunOperation::Sub, RunOperation::Mul, RunOperation::Div] {
        let shader_path = format!("src/gpu_acel/shader_code/{:?}.wgsl", op);
        let shader_code = fs::read_to_string(&shader_path)
            .expect(&format!("Failed to read shader file: {:?}", shader_path));
        cache.insert(op, shader_code);
    }

    cache
});

use wgpu::{Device, ShaderModuleDescriptor, ShaderSource, ShaderModule};

pub async fn compile_shader (device: &Device, shader_code: &str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Operation Shader"),
        source: ShaderSource::Wgsl(shader_code.into()),
    })
}