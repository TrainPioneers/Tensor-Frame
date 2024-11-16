use crate::Tensor;

impl<T> std::ops::Div for Tensor<T>
where
    T: util::ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64>,
{
    type Output = Tensor<T>;

    fn div(self, other: Self) -> Self::Output {
        todo!()
    }
}