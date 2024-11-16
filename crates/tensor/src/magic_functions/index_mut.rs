use crate::Tensor;

impl<T> std::ops::IndexMut<&[usize]> for Tensor<T>
where
    T: util::ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64> + Clone,
{
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        self.get_mut(index)
    }
}