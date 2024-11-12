use crate::Tensor;

impl<T> std::ops::Mul for Tensor<T>
where
    T: util::IsNum,
{
    type Output = Tensor<T>;

    fn mul(self, other: Self) -> Self::Output {
        todo!()
    }
}