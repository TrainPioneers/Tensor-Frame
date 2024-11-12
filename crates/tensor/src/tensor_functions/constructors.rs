use crate::tensor::Tensor;

impl Tensor {
    pub fn with_num(num: f32, shape: Vec<usize>) -> Tensor {
        let size = shape.iter().product();
        Tensor {
            data: vec![num; size],
            shape,
        }
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        Tensor::with_num(0., shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        Tensor::with_num(1., shape)
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        Tensor {
            data,
            shape
        }
    }
}