use std::error::Error;
use std::future::Future;

use futures::future::join_all;
use wgpu::{BufferSlice, Device, Queue};

use bind_group::*;
use buffer::*;
use encoder::*;
use pipeline::*;
use queue::*;
use receiver::*;
use setup::*;
use shader::*;
use util::{break_vec, ValidTensorType};
use workload::*;

mod buffer;
mod shader;
mod setup;
mod pipeline;
mod bind_group;
mod encoder;
mod queue;
mod workload;
mod receiver;

async fn run_vector_operation<T>(
    a: Vec<T>,
    b: Vec<T>,
    shader_code: &str,
) -> Result<Vec<T>, Box<dyn Error>>
where
    T: ValidTensorType,
{
    let setup: impl Future<Output=(Device, Queue)> = setup_wgpu();
    assert_eq!(a.len(), b.len(), "Incorrect shape");
    let t1_workload_future = break_vec(a.clone());
    let t2_workload_future = break_vec(b.clone());

    let (device, queue) = setup.await;
    let shader = compile_shader(&device,shader_code);

    let staging_buffer_future = create_staging_buffer(&device);

    let t1_workload = t1_workload_future.await;
    let length = t1_workload.len();
    let mut receivers_future: impl Future<Output=Vec<(Receiver,BufferSlice)>> = Vec::with_capacity(length);
    let t2_workload = t2_workload_future.await;
    let pipeline_future = create_pipeline(&device, shader.await);

    let pipeline = pipeline_future.await;
    let staging_buffer = staging_buffer_future.await;

    for i in 0..length {
        receivers_future[i] = create_workload(
            &device,
            &queue,
            &pipeline,
            &staging_buffer,
            &t1_workload[i],
            &t2_workload[i]
        );
    }

    let mut result: Vec<f32> = Vec::new();

    let receivers: Vec<(Receiver, BufferSlice)> = join_all(receivers_future).await;

    for i in receivers.iter(){
        let (receiver, buffer_slice) = i;
        if let Some(Ok(())) = receiver.receive() {
            let data = buffer_slice.get_mapped_range();
            let result_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            result.extend(result_data);
        } else {
            panic!("Failed to run compute on GPU!")
        }
    }
    staging_buffer.unmap();
    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {

    }
}
