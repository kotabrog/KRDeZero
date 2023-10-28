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
