use futures_intrusive::channel::shared::oneshot_channel;
use wgpu::{Buffer, BufferSlice, CommandEncoder, Device, Maintain, MapMode, Queue};

use crate::Receiver;

pub async fn submit_to_queue(
    device: &Device,
    queue: &Queue,
    mut encoder: CommandEncoder,
    staging_buffer: &Buffer,
) -> (Receiver,BufferSlice) {
    queue.submit(Some(encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = oneshot_channel();
    buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(Maintain::Wait);
    (receiver, buffer_slice)
}