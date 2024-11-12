use crate::Tensor;

impl<T> std::ops::Sub for Tensor<T>
where
    T: util::IsNum,
{
    type Output = Tensor<T>;

    fn sub(self, other: Self) -> Self::Output {
        todo!()
    }
}