use crate::Tensor;

impl<T> std::ops::Index<&[usize]> for Tensor<T>
where
    T: util::ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64> + Clone,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        self.get(index)
    }
}