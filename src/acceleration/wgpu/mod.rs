use std::future::Future;
use futures::future::join_all;
use wgpu::BufferSlice;
use crate::tensor::Tensor;

mod buffer;
mod shader;
mod setup;
mod pipeline;
mod bind_group;
mod encoder;
mod queue;
mod break_vec;
mod workload;
mod receiver;

use receiver::*;
use buffer::*;
use shader::*;
use setup::*;
use pipeline::*;
use bind_group::*;
use encoder::*;
use queue::*;
use break_vec::*;
use workload::*;
use crate::acceleration::run_functions::RunOperation;

pub async fn send_to_gpu(t1: Tensor, t2: Tensor, operation: RunOperation) -> Tensor{
    assert!(t1.same_shape(&t2), "Incorrect shape");
    let setup = setup_wgpu();
    let t1_workload_future = break_vec(t1.data.clone());
    let t2_workload_future = break_vec(t2.data.clone());
    let shader_code = SHADER_CACHE.get(&operation).unwrap().as_str();

    let (device, queue) = setup.await;

    let compiled_shader_future = compile_shader(&device, shader_code);
    let staging_buffer_future = create_staging_buffer(&device);

    let compiled_shader = compiled_shader_future.await;
    let pipeline_future = create_pipeline(&device, compiled_shader);

    let t1_workload = t1_workload_future.await;
    let length = t1_workload.len();
    let mut receivers_future: impl Future<Output=Vec<(Receiver,BufferSlice)>>= Vec::with_capacity(length);
    let t2_workload = t2_workload_future.await;

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

    let receivers: Vec<(Receiver, BufferSlice)> = join_all(receivers_future).await;

    let mut result: Vec<f32> = Vec::with_capacity(t1.data.len());

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
    Tensor::from_vec(result, t1.shape)
}