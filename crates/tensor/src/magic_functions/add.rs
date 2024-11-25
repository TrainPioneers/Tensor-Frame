use crate::Tensor;
use device_config::run_operation;
use util::RunOperation;

impl<T> std::ops::Add for Tensor<T>
where
    T: util::ValidTensorType + From<i32> + From<i64> + From<f32> + From<f64>,
{
    type Output = Tensor<T>;

    fn add(self, other: Self) -> Self::Output {
        let new_vec = run_operation::<T>(self.data, other.data, RunOperation::Add);
        Tensor::<T>::from_vec(new_vec, self.shape)
    }
}