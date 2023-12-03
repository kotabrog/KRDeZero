use super::{VariableData, VariableType};

impl VariableData {
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::None => &[],
            Self::F32(x) => x.get_shape(),
            Self::F64(x) => x.get_shape(),
            Self::I32(x) => x.get_shape(),
            Self::I64(x) => x.get_shape(),
            Self::USIZE(x) => x.get_shape(),
            Self::Bool(x) => x.get_shape(),
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            Self::None => 0,
            Self::F32(x) => x.ndim(),
            Self::F64(x) => x.ndim(),
            Self::I32(x) => x.ndim(),
            Self::I64(x) => x.ndim(),
            Self::USIZE(x) => x.ndim(),
            Self::Bool(x) => x.ndim(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::None => 0,
            Self::F32(x) => x.size(),
            Self::F64(x) => x.size(),
            Self::I32(x) => x.size(),
            Self::I64(x) => x.size(),
            Self::USIZE(x) => x.size(),
            Self::Bool(x) => x.size(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::F32(x) => x.len(),
            Self::F64(x) => x.len(),
            Self::I32(x) => x.len(),
            Self::I64(x) => x.len(),
            Self::USIZE(x) => x.len(),
            Self::Bool(x) => x.len(),
        }
    }

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

    pub fn get_variable_type(&self) -> VariableType {
        match self {
            Self::None => VariableType::None,
            Self::F32(_) => VariableType::F32,
            Self::F64(_) => VariableType::F64,
            Self::I32(_) => VariableType::I32,
            Self::I64(_) => VariableType::I64,
            Self::USIZE(_) => VariableType::USIZE,
            Self::Bool(_) => VariableType::Bool,
        }
    }
}
