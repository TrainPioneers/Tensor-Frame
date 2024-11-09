use futures_intrusive::channel::shared::OneshotReceiver;
use wgpu::BufferAsyncError;

pub type Receiver = OneshotReceiver<Result<(), BufferAsyncError>>;