use std::process::{Command, Output};
extern crate wgpu;
extern crate pollster;

pub fn has_cuda() -> bool {
    let output = Command::new("nvcc")
        .arg("--version")
        .output();

    process_output(output)
}

pub fn has_hip() -> bool {
    let output = Command::new("hipcc")
        .arg("--version")
        .output();

    process_output(output)
}

pub fn has_ocl() -> bool {
    let output = Command::new("clinfo")
        .output();

    process_output(output)
}

pub fn has_gpu() -> bool {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
    }));
    adapter.is_some()
}

fn process_output<E>(output: Result<Output, E>) -> bool {
    match output {
        Ok(output) => {
            if output.status.success() {
                true
            } else {
                false
            }
        }
        Err(_) => {
            false
        }
    }
}