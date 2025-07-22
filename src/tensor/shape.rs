use crate::error::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Result<Self> {
        // Validate that no dimension is zero (except for empty tensors)
        if dims.iter().any(|&dim| dim == 0) {
            return Err(crate::error::TensorError::InvalidShape(format!(
                "Shape dimensions cannot be zero: {:?}",
                dims
            )));
        }
        Ok(Shape { dims })
    }

    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    pub fn can_broadcast_to(&self, other: &Shape) -> bool {
        let self_dims = &self.dims;
        let other_dims = &other.dims;

        if self_dims.len() > other_dims.len() {
            return false;
        }

        let offset = other_dims.len() - self_dims.len();

        for (i, &self_dim) in self_dims.iter().enumerate() {
            let other_dim = other_dims[i + offset];
            if self_dim != 1 && self_dim != other_dim {
                return false;
            }
        }

        true
    }

    pub fn broadcast_shape(&self, other: &Shape) -> Option<Shape> {
        let self_dims = &self.dims;
        let other_dims = &other.dims;

        let max_len = self_dims.len().max(other_dims.len());
        let mut result_dims = vec![1; max_len];

        for i in 0..max_len {
            let self_dim = if i < self_dims.len() {
                self_dims[self_dims.len() - 1 - i]
            } else {
                1
            };

            let other_dim = if i < other_dims.len() {
                other_dims[other_dims.len() - 1 - i]
            } else {
                1
            };

            if self_dim == 1 {
                result_dims[max_len - 1 - i] = other_dim;
            } else if other_dim == 1 || self_dim == other_dim {
                result_dims[max_len - 1 - i] = self_dim;
            } else {
                return None; // Incompatible shapes
            }
        }

        Some(Shape { dims: result_dims })
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims).unwrap_or_else(|_| {
            // For backwards compatibility, create empty tensor shape for invalid shapes
            // This prevents panics but should be avoided in new code
            Shape { dims: vec![0] }
        })
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec()).unwrap_or_else(|_| {
            // For backwards compatibility, create empty tensor shape for invalid shapes
            // This prevents panics but should be avoided in new code
            Shape { dims: vec![0] }
        })
    }
}
