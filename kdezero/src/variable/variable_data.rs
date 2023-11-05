mod info;
mod create;
mod conversion;
mod operater;

use ktensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub enum VariableData {
    None,
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    USIZE(Tensor<usize>),
    Bool(Tensor<bool>),
}

impl std::fmt::Display for VariableData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::F32(x) => write!(f, "{}", x),
            Self::F64(x) => write!(f, "{}", x),
            Self::I32(x) => write!(f, "{}", x),
            Self::I64(x) => write!(f, "{}", x),
            Self::USIZE(x) => write!(f, "{}", x),
            Self::Bool(x) => write!(f, "{}", x),
        }
    }
}
