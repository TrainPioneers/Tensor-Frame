
#[cfg(test)]
mod tests {
    #[test]
    fn add_1() {
        assert_eq!(1, 1);
    }
}

pub mod prelude {
    pub use tensor::Tensor;
    pub use device_config::run_operation;
    pub use util::RunOperation;
}

pub mod dev {

}