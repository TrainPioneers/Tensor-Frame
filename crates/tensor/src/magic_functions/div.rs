use crate::Tensor;

impl<T> std::ops::Div for Tensor<T>
where
    T: util::ValidTensorType,
{
    type Output = Tensor<T>;

    fn div(self, other: Self) -> Self::Output {
        todo!()
    }
}