use crate::tensor::Tensor;

impl std::ops::IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        todo!()
    }
}