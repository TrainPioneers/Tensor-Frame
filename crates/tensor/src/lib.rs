use util::ValidTensorType;

mod magic_functions;
mod tensor_functions;
#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T>
where
    T: ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64>,
{
    data: Vec<T>,
    shape: Vec<usize>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
