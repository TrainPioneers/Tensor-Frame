use wgpu::{
    Backends,
    Device,
    DeviceDescriptor,
    Instance,
    InstanceDescriptor,
    Queue,
    RequestAdapterOptions
};

pub async fn setup_wgpu() -> (Device, Queue) {
    // Create an instance of wgpu
    let instance = Instance::new(InstanceDescriptor{
        backends:Backends::all(),
        flags: Default::default(),
        dx12_shader_compiler: Default::default(),
        gles_minor_version: Default::default(),
    });

    // Request an adapter that supports the features you need
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            compatible_surface: None,
            ..Default::default()
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Request a device and queue
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("Device"),
            required_features: Default::default(),
            required_limits: Default::default(),
            memory_hints: Default::default(),
        }, None)
        .await
        .expect("Failed to create device");

    (device, queue)
}