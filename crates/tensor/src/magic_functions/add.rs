use crate::Tensor;

impl<T> std::ops::Add for Tensor<T>
where
    T: util::IsNum,
{
    type Output = Tensor<T>;

    fn add(self, other: Self) -> Self::Output {
        todo!()
    }
}