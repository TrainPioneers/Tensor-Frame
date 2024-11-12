use wgpu::{ComputePipeline, Device, Queue, Buffer, BufferSlice};
use crate::acceleration::wgpu::{Receiver,create_buffer, create_result_buffer, create_bind_group, run_encoder, submit_to_queue, create_encoder};

pub async fn create_workload(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    staging_buffer: &Buffer,
    d1: &Vec<f32>,
    d2: &Vec<f32>
) -> (Receiver,BufferSlice) {
    let size = (d1.len() * std::mem::size_of::<f32>()) as u64;
    let encoder_future = create_encoder(&device);
    let buffer_a_future = create_buffer(&device, d1.clone());
    let buffer_b_future = create_buffer(&device, d2.clone());
    let result_buffer_future = create_result_buffer(&device, size);

    let layout = pipeline.get_bind_group_layout(0);
    let buffer_a = buffer_a_future.await;
    let buffer_b = buffer_b_future.await;
    let result_buffer = result_buffer_future.await;

    let bind_group = create_bind_group(&device, layout, buffer_a, buffer_b, result_buffer).await;

    let mut encoder = encoder_future.await;

    run_encoder(&mut encoder, pipeline, bind_group, d1.len()).await;

    submit_to_queue(device, queue, encoder, staging_buffer)
}