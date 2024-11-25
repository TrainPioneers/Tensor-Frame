use crate::Tensor;

impl<T> Tensor<T>
where
    T: util::ValidTensorType + Clone,
{
    // Helper method to calculate the flat index from multidimensional indices
    fn flatten_index(&self, indices: &[usize]) -> usize {
        indices.iter()
            .zip(&self.shape)
            .fold(0, |acc, (idx, dim)| acc * dim + idx)
    }
    pub fn same_shape(&self, other: &Tensor<T>) -> bool
    where
        T: util::ValidTensorType,
    {
        self.shape == other.shape
    }
    // Get an element at the specified multidimensional index
    pub fn get(&self, indices: &[usize]) -> &T {
        assert_eq!(indices.len(), self.shape.len(), "Incorrect number of indices");
        let index = self.flatten_index(indices);
        &self.data[index]
    }

    // Set an element at the specified multidimensional index
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        assert_eq!(indices.len(), self.shape.len(), "Incorrect number of indices");
        let index = self.flatten_index(indices);
        self.data.get_mut(index).unwrap()
    }
}