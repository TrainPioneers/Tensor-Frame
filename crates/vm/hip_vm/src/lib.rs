use libloading::{Library, Symbol};
use util::ValidTensorType;
use std::error::Error;

fn run_vector_operation<T>(a: Vec<T>, b: Vec<T>, function: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: ValidTensorType,
{
    // Load the shared library containing the HIP kernel
    // Check that a and b are the same length
    if a.len() != b.len() {
        return Err(Box::from("Input vectors must have the same length"));
    }

    let num_elements = a.len() as i32;
    let mut result = vec![T::default(); a.len()];

    unsafe {
        let lib = Library::new("target/debug/libvector_math.so")?;

        // Define the kernel function signature
        let kernel_func: Symbol<unsafe extern "C" fn(*const T, *const T, *mut T, i32)>;

        // Match the function string to load the corresponding kernel function
        match function {
            "add" => {
                kernel_func = unsafe { lib.get(b"add")? };
            },
            "sub" => {
                kernel_func = unsafe { lib.get(b"sub")? };
            },
            "mul" => {
                kernel_func = unsafe { lib.get(b"mul")? };
            },
            "div" => {
                kernel_func = unsafe { lib.get(b"div")? };
            },
            _ => {
                return Err(Box::from("Invalid function name. Use 'add', 'sub', 'mul', or 'div'"));
            }
        }


        // Execute the kernel on the GPU

        kernel_func(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), num_elements);
        return Ok(result)
    }
}
