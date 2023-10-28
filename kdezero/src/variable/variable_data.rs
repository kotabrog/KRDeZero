mod operate;

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

impl VariableData {
    pub fn data_type(&self) -> &str {
        match self {
            Self::None => "None",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::USIZE(_) => "usize",
            Self::Bool(_) => "bool",
        }
    }
}

impl From<f32> for VariableData {
    fn from(data: f32) -> Self {
        Self::F32(Tensor::scalar(data))
    }
}

impl From<f64> for VariableData {
    fn from(data: f64) -> Self {
        Self::F64(Tensor::scalar(data))
    }
}

impl From<i32> for VariableData {
    fn from(data: i32) -> Self {
        Self::I32(Tensor::scalar(data))
    }
}

impl From<i64> for VariableData {
    fn from(data: i64) -> Self {
        Self::I64(Tensor::scalar(data))
    }
}

impl From<usize> for VariableData {
    fn from(data: usize) -> Self {
        Self::USIZE(Tensor::scalar(data))
    }
}

impl From<bool> for VariableData {
    fn from(data: bool) -> Self {
        Self::Bool(Tensor::scalar(data))
    }
}

impl From<Tensor<f32>> for VariableData {
    fn from(data: Tensor<f32>) -> Self {
        Self::F32(data)
    }
}

impl From<Tensor<f64>> for VariableData {
    fn from(data: Tensor<f64>) -> Self {
        Self::F64(data)
    }
}

impl From<Tensor<i32>> for VariableData {
    fn from(data: Tensor<i32>) -> Self {
        Self::I32(data)
    }
}

impl From<Tensor<i64>> for VariableData {
    fn from(data: Tensor<i64>) -> Self {
        Self::I64(data)
    }
}

impl From<Tensor<usize>> for VariableData {
    fn from(data: Tensor<usize>) -> Self {
        Self::USIZE(data)
    }
}

impl From<Tensor<bool>> for VariableData {
    fn from(data: Tensor<bool>) -> Self {
        Self::Bool(data)
    }
}
