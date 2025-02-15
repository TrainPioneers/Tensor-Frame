// This tensor class has a flexible shape and works with i32, i64, f32, and f64.

// T is the type of the data in the tensor. Limit T to i32, i64, f32, and f64.
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div};

pub trait TensorElement: Copy + Debug + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {}

impl TensorElement for i32 {}
impl TensorElement for i64 {}
impl TensorElement for f32 {}
impl TensorElement for f64 {}


pub struct Tensor<T> where T: TensorElement {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T> where T: TensorElement {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert!(data.len() == shape.iter().product(), "The size of the data does not match the shape.");
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(vec![T::default(); shape.iter().product()], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        Self::new(vec![T::one(); shape.iter().product()], shape)
    }

    pub fn random(shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        Self::new(vec![T::default(); shape.iter().product()], shape)
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert!(self.data.len() == shape.iter().product(), "The size of the data does not match the shape.");
        Self { data: self.data, shape }
    }
}

impl<T> Index<&[usize]> for Tensor<T> where T: TensorElement {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.data[index.iter().product::<usize>()]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> where T: TensorElement {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        &mut self.data[index.iter().product::<usize>()]
    }
}

impl<T> Add for Tensor<T> where T: TensorElement {
    type Output = Tensor<T>;

    fn add(self, other: Self) -> Self::Output {
        Self::new(self.data + other.data, self.shape)
    }
}

impl<T> Sub for Tensor<T> where T: TensorElement {
    type Output = Tensor<T>;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.data - other.data, self.shape)
    }
}   

impl<T> Mul for Tensor<T> where T: TensorElement {
    type Output = Tensor<T>;

    fn mul(self, other: Self) -> Self::Output {
        Self::new(self.data * other.data, self.shape)   
    }
}

impl<T> Div for Tensor<T> where T: TensorElement {
    type Output = Tensor<T>;

    fn div(self, other: Self) -> Self::Output {
        Self::new(self.data / other.data, self.shape)
    }
}
