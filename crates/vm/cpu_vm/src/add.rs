use std::ops::{Add, Div, Mul, Sub};

use rayon::iter::IntoParallelRefIterator;

use util::ValidTensorType;

fn vector_add<T>(a: Vec<T>, b: Vec<T>) -> Vec<T>
where
    T: ValidTensorType + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Clone + Send + Sync,
{
    a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| x.clone() + y.clone())
        .collect()
}
