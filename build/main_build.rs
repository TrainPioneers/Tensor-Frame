use std::process::{Command, exit, Output};

fn main() {
    let prefer_cpu =  std::env::var("CARGO_FEATURE_PREFER_CPU_OVER_WGPU").is_ok();
    let feature = if has_cuda() {
        "cuda"
    } else if has_hip(){
        "hip"
    } else if has_ocl() {
        "ocl"
    } else if has_gpu() && !prefer_cpu {
        "wgpu"
    } else {
        "cpu"
    };

    println!("cargo:rustc-cfg=device_config_{}", feature);
}


fn has_cuda() -> bool{
    let output = Command::new("nvcc")
        .arg("--version")
        .output();

    process_output(output)
}

fn has_hip() -> bool{
    let output = Command::new("hipcc")
        .arg("--version")
        .output();

    process_output(output)
}

fn has_ocl() -> bool{
    let output = Command::new("clinfo")
        .output();

    process_output(output)
}

fn has_gpu() -> bool {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
    }));
    adapter.is_some()
}

fn process_output<E>(output: Result<Output, E>) -> bool{
    match output {
        Ok(output) => {
            if output.status.success() {
                true
            } else {
                false
            }
        },
        Err(_) => {
            false
        }
    }
}