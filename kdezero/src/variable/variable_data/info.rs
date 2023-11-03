use super::VariableData;

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
