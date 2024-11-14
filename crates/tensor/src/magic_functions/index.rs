use crate::Tensor;

impl<T> std::ops::Index<&[usize]> for Tensor<T>
where
    T: util::ValidTensorType + Clone,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        todo!();
        &self.get(index)
    }
}