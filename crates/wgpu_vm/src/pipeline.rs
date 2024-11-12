use std::collections::HashMap;
use wgpu::{Device, ShaderModule, ComputePipelineDescriptor, PipelineCompilationOptions, ComputePipeline};
pub async fn create_pipeline(device: &Device, shader_module: ShaderModule) -> ComputePipeline {
    let compilation_options = PipelineCompilationOptions {
        constants: &HashMap::new(),
        zero_initialize_workgroup_memory: false,
    };

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options,
        cache: None
    })
}