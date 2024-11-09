use crate::tensor::Tensor;

impl std::ops::Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        todo!()
    }
}