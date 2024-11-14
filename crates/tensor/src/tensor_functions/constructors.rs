use crate::Tensor;

impl<T> Tensor<T>
where
    T: util::ValidTensorType,
{
    pub fn with_num(num: T, shape: Vec<usize>) -> Tensor<T> {
        let size = shape.iter().product();
        Tensor {
            data: vec![num; size],
            shape,
        }
    }
    pub fn zeros(shape: Vec<usize>) -> Tensor<T> {
        Tensor::with_num(0 as T, shape)
    }

    pub fn ones(shape: Vec<usize>) -> Tensor<T> {
        Tensor::with_num(1 as T, shape)
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
        assert_eq!(data.len(), shape.iter().product());
        Tensor {
            data,
            shape
        }
    }
}