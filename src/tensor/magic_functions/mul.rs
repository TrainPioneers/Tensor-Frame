use crate::tensor::Tensor;

impl std::ops::Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        todo!()
    }
}