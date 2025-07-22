use crate::error::Result;
use crate::tensor::shape::Shape;
use once_cell::sync::Lazy;
use std::fmt::Debug;
use std::sync::Arc;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub trait Backend: Debug + Send + Sync {
    fn is_available(&self) -> bool {
        true
    }

    fn zeros(&self, shape: &Shape) -> Result<Storage>;
    fn ones(&self, shape: &Shape) -> Result<Storage>;
    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage>;

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;
    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;

    fn sum(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage>;
    fn mean(&self, storage: &Storage, axis: Option<usize>) -> Result<Storage>;
    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage>;

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>>;

    fn matmul(&self, lhs: &Storage, rhs: &Storage, lhs_shape: &Shape, rhs_shape: &Shape) -> Result<Storage>;
}

#[derive(Debug, Clone)]
pub enum Storage {
    #[cfg(feature = "cpu")]
    Cpu(Vec<f32>),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "wgpu")]
    Wgpu(wgpu::WgpuStorage),
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaStorage {
    pub buffer: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
}

pub static BACKENDS: Lazy<Vec<Arc<dyn Backend>>> = Lazy::new(|| {
    let mut backends: Vec<Arc<dyn Backend>> = Vec::new();
    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        if let Ok(backend) = cuda::CudaBackend::new() {
            backends.push(Arc::new(backend) as Arc<dyn Backend>);
        }
    }
    #[cfg(feature = "wgpu")]
    if wgpu::is_available() {
        if let Ok(backend) = wgpu::WgpuBackend::new_blocking() {
            backends.push(Arc::new(backend) as Arc<dyn Backend>);
        }
    }
    #[cfg(feature = "cpu")]
    {
        backends.push(Arc::new(cpu::CpuBackend::new()) as Arc<dyn Backend>);
    }

    backends
});
