use anyhow::Result;
use ktensor::Tensor;
use ktensor::tensor::TensorRng;
use crate::error::KDeZeroError;
use super::{VariableData, VariableType};

impl VariableData {
    pub fn none() -> Self {
        Self::None
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Ok(match self {
            Self::F32(x) => Self::F32(Tensor::zeros_like(x)),
            Self::F64(x) => Self::F64(Tensor::zeros_like(x)),
            Self::I32(x) => Self::I32(Tensor::zeros_like(x)),
            Self::I64(x) => Self::I64(Tensor::zeros_like(x)),
            Self::USIZE(x) => Self::USIZE(Tensor::zeros_like(x)),
            _ => return Err(KDeZeroError::NotImplementedType(
                self.data_type().to_string(),
                "zeros_like".to_string(),
            ).into()),
        })
    }

    pub fn zeros_type(shape: &[usize], variable_type: VariableType) -> Result<Self> {
        Ok(match variable_type {
            VariableType::F32 => Self::F32(Tensor::zeros(shape)),
            VariableType::F64 => Self::F64(Tensor::zeros(shape)),
            VariableType::I32 => Self::I32(Tensor::zeros(shape)),
            VariableType::I64 => Self::I64(Tensor::zeros(shape)),
            VariableType::USIZE => Self::USIZE(Tensor::zeros(shape)),
            _ => return Err(KDeZeroError::NotImplementedType(
                variable_type.to_string(),
                "zeros_type".to_string(),
            ).into()),
        })
    }

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

    pub fn random_normal(shape: &[usize], variable_type: VariableType) -> Result<Self> {
        let mut rng = TensorRng::new();
        Ok(match variable_type {
            VariableType::F32 => Self::F32(rng.normal(shape, 0.0, 1.0)?),
            VariableType::F64 => Self::F64(rng.normal(shape, 0.0, 1.0)?),
            _ => return Err(KDeZeroError::NotImplementedType(
                variable_type.to_string(),
                "normal".to_string(),
            ).into()),
        })
    }

    pub fn eye_like_type(&self, n: usize) -> Result<Self> {
        Ok(match self {
            Self::F32(_) => Self::F32(Tensor::eye(n)),
            Self::F64(_) => Self::F64(Tensor::eye(n)),
            Self::I32(_) => Self::I32(Tensor::eye(n)),
            Self::I64(_) => Self::I64(Tensor::eye(n)),
            Self::USIZE(_) => Self::USIZE(Tensor::eye(n)),
            _ => return Err(KDeZeroError::NotImplementedType(
                self.data_type().to_string(),
                "eye_like".to_string(),
            ).into()),
        })
    }
}
