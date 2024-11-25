use device_config::run_operation;
use util::RunOperation;
use crate::Tensor;

impl<T> std::ops::Sub for Tensor<T>
where
    T: util::ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64>,
{
    type Output = Tensor<T>;

    fn sub(self, other: Self) -> Self::Output {
        let new_vec = run_operation::<T>(self.data, other.data, RunOperation::Sub);
        Tensor::<T>::from_vec(new_vec, self.shape)
    }
}