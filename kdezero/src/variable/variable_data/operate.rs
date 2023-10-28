use anyhow::Result;
use super::VariableData;
use crate::error::KDeZeroError;

impl VariableData {
    pub fn square(&self) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => (x * x).into(),
            VariableData::F64(x) => (x * x).into(),
            VariableData::I32(x) => (x * x).into(),
            VariableData::I64(x) => (x * x).into(),
            VariableData::USIZE(x) => (x * x).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "square".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn exp(&self) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.exp().into(),
            VariableData::F64(x) => x.exp().into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "exp".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn sub(&self, other: &VariableData) -> Result<VariableData> {
        Ok(match (self, other) {
            (VariableData::F32(x), VariableData::F32(y)) => (x - y).into(),
            (VariableData::F64(x), VariableData::F64(y)) => (x - y).into(),
            (VariableData::I32(x), VariableData::I32(y)) => (x - y).into(),
            (VariableData::I64(x), VariableData::I64(y)) => (x - y).into(),
            (VariableData::USIZE(x), VariableData::USIZE(y)) => (x - y).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "sub".to_string(),
                format!("{:?}, {:?}", self.data_type(), other.data_type()),
            ).into()),
        })
    }

    pub fn scalar_add(&self, value: f64) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => (x + value as f32).into(),
            VariableData::F64(x) => (x + value).into(),
            VariableData::I32(x) => (x + value as i32).into(),
            VariableData::I64(x) => (x + value as i64).into(),
            VariableData::USIZE(x) => (x + value as usize).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "scalar_add".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn scalar_mul(&self, value: f64) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => (x * value as f32).into(),
            VariableData::F64(x) => (x * value).into(),
            VariableData::I32(x) => (x * value as i32).into(),
            VariableData::I64(x) => (x * value as i64).into(),
            VariableData::USIZE(x) => (x * value as usize).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "scalar_mul".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.square()?;
        assert_eq!(y, VariableData::from(4.0f32));
        Ok(())
    }

    #[test]
    fn error_square_bool() -> Result<()> {
        let x = VariableData::from(true);
        match x.square() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "square".to_string(),
                    x.data_type().to_string(),
                ));
            }
        }
        Ok(())
    }
}
