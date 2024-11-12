use crate::Tensor;

impl<T> std::ops::Index<&[usize]> for Tensor<T>
where
    T: util::IsNum,
{
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        self.get(index)
    }
}