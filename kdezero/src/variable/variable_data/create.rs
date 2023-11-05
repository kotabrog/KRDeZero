use anyhow::Result;
use ktensor::Tensor;
use crate::error::KDeZeroError;
use super::VariableData;

impl VariableData {
    pub fn ones_like(&self) -> Result<Self> {
        Ok(match self {
            Self::F32(x) => Self::F32(Tensor::ones_like(x)),
            Self::F64(x) => Self::F64(Tensor::ones_like(x)),
            Self::I32(x) => Self::I32(Tensor::ones_like(x)),
            Self::I64(x) => Self::I64(Tensor::ones_like(x)),
            Self::USIZE(x) => Self::USIZE(Tensor::ones_like(x)),
            _ => return Err(KDeZeroError::NotImplementedType(
                self.data_type().to_string(),
                "ones_like".to_string(),
            ).into()),
        })
    }

    pub fn full_like(&self, value: f64) -> Result<Self> {
        Ok(match self {
            Self::F32(x) => Self::F32(Tensor::full_like(value as f32, x)),
            Self::F64(x) => Self::F64(Tensor::full_like(value, x)),
            Self::I32(x) => Self::I32(Tensor::full_like(value as i32, x)),
            Self::I64(x) => Self::I64(Tensor::full_like(value as i64, x)),
            Self::USIZE(x) => Self::USIZE(Tensor::full_like(value as usize, x)),
            _ => return Err(KDeZeroError::NotImplementedType(
                self.data_type().to_string(),
                "full_like".to_string(),
            ).into()),
        })
    }
}
