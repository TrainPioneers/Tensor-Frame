use crate::tensor::Tensor;

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.same_shape(other)
    }
}