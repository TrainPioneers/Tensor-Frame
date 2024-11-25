pub trait ValidTensorType: Sized {
    fn to_self(value: i32) -> Self;
}

impl ValidTensorType for i32 {
    fn to_self(value: i32) -> Self {
        value
    }
}
impl ValidTensorType for f32 {
    fn to_self(value: i32) -> Self {
        value as f32
    }
}

impl ValidTensorType for i64 {
    fn to_self(value: i32) -> Self {
        value as i64
    }
}

impl ValidTensorType for f64 {
    fn to_self(value: i32) -> Self {
        value as f64
    }
}