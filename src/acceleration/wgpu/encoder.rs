use wgpu::{Device, CommandEncoder, ComputePipeline, BindGroup,CommandEncoderDescriptor,ComputePassDescriptor};

pub async fn create_encoder (device: &Device) -> CommandEncoder {
    device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Operation Encoder"),
    })
}

pub async fn run_encoder(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: BindGroup,
    data_len: usize
){
    let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    let workgroup_count = ((data_len as u32) + 63) / 64;
    cpass.dispatch_workgroups(workgroup_count, 1, 1);
}