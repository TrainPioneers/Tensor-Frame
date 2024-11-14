use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

pub async fn compile_shader (device: &Device, shader_code: &str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Operation Shader"),
        source: ShaderSource::Wgsl(shader_code.into()),
    })
}