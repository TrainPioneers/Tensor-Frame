use util::ValidTensorType;

mod magic_functions;
mod tensor_functions;
#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T = i32>
where
    T: ValidTensorType,
{
    data: Vec<T>,
    shape: Vec<usize>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
