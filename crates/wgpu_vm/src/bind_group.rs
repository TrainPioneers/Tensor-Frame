use wgpu::{BindGroupLayout, Buffer, Device, BindGroupDescriptor, BindGroupEntry, BindGroup};
pub async fn create_bind_group (
    device: &Device,
    bind_group_layout: BindGroupLayout,
    a: Buffer,
    b: Buffer,
    c: Buffer
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: c.as_entire_binding(),
            },
        ],
        label: Some("Bind Group"),
    })
}