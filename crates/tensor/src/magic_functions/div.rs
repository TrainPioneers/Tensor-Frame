use device_config::run_operation;
use util::RunOperation;
use crate::Tensor;

impl<T> std::ops::Div for Tensor<T>
where
    T: util::ValidTensorType + Clone + Copy + Default,
{
    type Output = Tensor<T>;

    fn div(self, other: Self) -> Self::Output {
        let new_vec = run_operation::<T>(self.data, other.data, RunOperation::Div);
        Tensor::<T>::from_vec(new_vec, self.shape)
    }
}