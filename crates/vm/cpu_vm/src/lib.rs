use std::error::Error;

use util::ValidTensorType;

mod add;
mod mul;
mod sub;
mod div;
use add::*;
use mul::*;
use sub::*;
use div::*;

pub fn run_vector_operation<T>(
    a: Vec<T>,
    b: Vec<T>,
    function_name: &str,
) -> Result<Vec<T>, Box<dyn Error>>
where
    T: ValidTensorType,
{
    Ok(
        match function_name {
            "add" => vector_add(a, b),
            "sub" => vector_sub(a, b),
            "mul.cl" => vector_mul(a, b),
            "div" => vector_div(a, b),
            _ => panic!("Incorrect function name")
        }
    )
}
