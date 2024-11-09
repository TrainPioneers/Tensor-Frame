mod magic_functions;
mod tensor_functions;

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Tensor {
    pub data: Vec<f32>,
    pub(crate) shape: Vec<usize>
}
