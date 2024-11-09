use crate::tensor::Tensor;

impl Tensor {
    // Helper method to calculate the flat index from multidimensional indices
    fn flatten_index(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(&self.shape)
            .fold(0, |acc, (idx, dim)| acc * dim + idx)
    }
    pub fn same_shape(&self, other: &Tensor) -> bool{
        self.shape == other.shape
    }
}
