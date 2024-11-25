
#[cfg(test)]
mod tests {
    #[test]
    fn equals() {
        use tensor::Tensor;
        assert_eq!(Tensor::from_vec(vec![1, 2, 3, 4, 5], vec![5]), Tensor::from_vec(vec![1, 2, 3, 4, 5], vec![5]));
    }
    #[test]
    fn add() {
        use tensor::Tensor;
        let t1 = Tensor::<f32>::zeros(vec![5]);
        let t2 = Tensor::<f32>::ones(vec![5]);
        assert_eq!(t1 + t2, Tensor::<f32>::ones(vec![5]));
    }
}

pub mod prelude {
    pub use tensor::Tensor;
    pub use device_config::run_operation;
    pub use util::RunOperation;
}

pub mod dev {

}