use crate::Tensor;

impl<T> std::ops::IndexMut<&[usize]> for Tensor<T>
where
    T: util::IsNum,
{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        self.get_mut(index)
    }
}