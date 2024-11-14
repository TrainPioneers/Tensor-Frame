use crate::Tensor;

impl<T> std::ops::Sub for Tensor<T>
where
    T: util::ValidTensorType,
{
    type Output = Tensor<T>;

    fn sub(self, other: Self) -> Self::Output {
        todo!()
    }
}