use crate::Tensor;

impl<T> PartialEq for Tensor<T>
where
    T: util::IsNum,
{
    fn eq(&self, other: &Self) -> bool {
        /*self.data == other.data && */ self.same_shape(other)
    }
}