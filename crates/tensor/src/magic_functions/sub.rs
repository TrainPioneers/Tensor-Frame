use crate::Tensor;

impl<T> std::ops::Sub for Tensor<T>
where
    T: util::ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64>,
{
    type Output = Tensor<T>;

    fn sub(self, other: Self) -> Self::Output {
        todo!()
    }
}