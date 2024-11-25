use device_config::run_operation;
use util::RunOperation;
use crate::Tensor;

impl<T> std::ops::Mul for Tensor<T>
where
    T: util::ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64>,
{
    type Output = Tensor<T>;

    fn mul(self, other: Self) -> Self::Output {
        let new_vec = run_operation::<T>(self.data, other.data, RunOperation::Mul);
        Tensor::<T>::from_vec(new_vec, self.shape)
    }
}