use std::{env, process::Command};

fn main() {
    let mut has_nvidia = env::var("CARGO_FEATURE_CUDA").is_ok();
    let mut has_wgpu = env::var("CARGO_FEATURE_WGPU").is_ok();

    let nvidia_present = Command::new("nvidia-smi")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);


    if has_nvidia && has_wgpu {
        panic!("You cannot enable both 'cuda' and 'wgpu' features at the same time.");
    }
    else if !(has_nvidia && has_wgpu){
        if nvidia_present {
            println!("cargo:rustc-cfg=feature=\"cuda\"");
            has_nvidia = true;
        } else {
            println!("cargo:rustc-cfg=feature=\"wgpu\"");
            has_wgpu = true;
        }
    }

    if has_wgpu {
        println!("No compilation options available.")
    } else if has_nvidia {
        // Check if the CUDA compiler (nvcc) is available in the environment
        let nvcc_check = Command::new("nvcc")
            .arg("--version")
            .output();

        match nvcc_check {
            Ok(output) if output.status.success() => {
                // CUDA is available, proceed with the compilation to SASS
                compile_cuda_to_sass();
            }
            _ => {
                println!("cargo:warning=CUDA not available. Skipping CUDA compilation.");
                println!("cargo:rustc-cfg=feature=\"cpu\"")
            }
        }
    } else {
        panic!("Something didn't work while building the project")
    }


}
fn compile_cuda_to_sass() {
    // Path to the CUDA compiler (nvcc)
    let nvcc_path = "nvcc"; // Assuming it's in the PATH

    // Path to the kernel file (adjust as needed)
    let kernel_path = "instructions/cuda/kernel.cu"; // Modify this path based on where you store your .cu file

    // Determine the target directory based on the build profile (debug or release)
    let target_dir = if cfg!(debug_assertions) {
        "target/debug"  // For debug builds
    } else {
        "target/release" // For release builds
    };

    env::set_var("DIR",target_dir);

    // Output path for the compiled SASS (binary object file)
    let output_sass = format!("{}/kernel.sass", target_dir);

    // Run the nvcc command to compile the .cu file into SASS (binary)
    let status = Command::new(nvcc_path)
        .arg("-c")         // Compile to binary code (SASS)
        .arg(kernel_path)  // Path to .cu file
        .arg("-o")
        .arg(&output_sass)   // Output SASS file
        .status()
        .expect("Failed to execute nvcc");

    // Ensure the command succeeded
    if !status.success() {
        panic!("CUDA compilation failed");
    }

    // Tell cargo to re-run build script if the kernel file changes
    println!("cargo:rerun-if-changed={}", kernel_path);
}
