pub trait ValidTensorType {}

impl ValidTensorType for i32 {}
impl ValidTensorType for i64 {}
impl ValidTensorType for f32 {}
impl ValidTensorType for f64 {}