use ocl::{Buffer, ProQue};
use std::error::Error;
use util::ValidTensorType;

fn run_vector_operation<T>(a: Vec<T>, b: Vec<T>, function: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: ValidTensorType + ocl::OclPrm,
{
    // Ensure the vectors are the same length
    if a.len() != b.len() {
        return Err("Vectors must have the same length".into());
    }

    // Set up OpenCL context, queue, and program with preloaded kernel code
    let pro_que = ProQue::builder()
        .src(function) // Use the preloaded kernel code
        .dims(a.len())
        .build()?;

    // Create buffers for inputs and outputs
    let buffer_a = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(a.len())
        .copy_host_slice(&a)
        .build()?;

    let buffer_b = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(b.len())
        .copy_host_slice(&b)
        .build()?;

    let buffer_result = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(a.len())
        .build()?;

    // Build kernel
    let kernel = pro_que
        .kernel_builder("vector_op") // Ensure this matches the kernel function name in `function`
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_result)
        .build()?;

    // Run kernel
    unsafe {
        kernel.enq()?;
    }

    // Read result back to host
    let mut result = vec![T::default(); a.len()];
    buffer_result.read(&mut result).enq()?;

    Ok(result)
}
