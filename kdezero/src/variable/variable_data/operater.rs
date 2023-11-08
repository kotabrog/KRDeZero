use anyhow::Result;
use super::VariableData;
use crate::error::KDeZeroError;

impl VariableData {
    pub fn add(&self, other: &VariableData) -> Result<VariableData> {
        Ok(match (self, other) {
            (VariableData::F32(x), VariableData::F32(y)) => (x + y).into(),
            (VariableData::F64(x), VariableData::F64(y)) => (x + y).into(),
            (VariableData::I32(x), VariableData::I32(y)) => (x + y).into(),
            (VariableData::I64(x), VariableData::I64(y)) => (x + y).into(),
            (VariableData::USIZE(x), VariableData::USIZE(y)) => (x + y).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "add".to_string(),
                format!("{:?}, {:?}", self.data_type(), other.data_type()),
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

    pub fn mul(&self, other: &VariableData) -> Result<VariableData> {
        Ok(match (self, other) {
            (VariableData::F32(x), VariableData::F32(y)) => (x * y).into(),
            (VariableData::F64(x), VariableData::F64(y)) => (x * y).into(),
            (VariableData::I32(x), VariableData::I32(y)) => (x * y).into(),
            (VariableData::I64(x), VariableData::I64(y)) => (x * y).into(),
            (VariableData::USIZE(x), VariableData::USIZE(y)) => (x * y).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "mul".to_string(),
                format!("{:?}, {:?}", self.data_type(), other.data_type()),
            ).into()),
        })
    }

    pub fn div(&self, other: &VariableData) -> Result<VariableData> {
        Ok(match (self, other) {
            (VariableData::F32(x), VariableData::F32(y)) => (x / y).into(),
            (VariableData::F64(x), VariableData::F64(y)) => (x / y).into(),
            (VariableData::I32(x), VariableData::I32(y)) => (x / y).into(),
            (VariableData::I64(x), VariableData::I64(y)) => (x / y).into(),
            (VariableData::USIZE(x), VariableData::USIZE(y)) => (x / y).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "div".to_string(),
                format!("{:?}, {:?}", self.data_type(), other.data_type()),
            ).into()),
        })
    }

    pub fn neg(&self) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => (-x).into(),
            VariableData::F64(x) => (-x).into(),
            VariableData::I32(x) => (-x).into(),
            VariableData::I64(x) => (-x).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "neg".to_string(),
                self.data_type().to_string(),
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

    pub fn pow(&self, n: f64) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.powf(n as f32).into(),
            VariableData::F64(x) => x.powf(n).into(),
            VariableData::I32(x) => x.pow(n as u32).into(),
            VariableData::I64(x) => x.pow(n as u32).into(),
            VariableData::USIZE(x) => x.pow(n as u32).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "pow".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn sin(&self) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.sin().into(),
            VariableData::F64(x) => x.sin().into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "sin".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn cos(&self) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.cos().into(),
            VariableData::F64(x) => x.cos().into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "cos".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn tanh(&self) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.tanh().into(),
            VariableData::F64(x) => x.tanh().into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "tanh".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn sum(&self, axis: Option<&Vec<usize>>, keepdims: bool) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.sum(axis, keepdims).into(),
            VariableData::F64(x) => x.sum(axis, keepdims).into(),
            VariableData::I32(x) => x.sum(axis, keepdims).into(),
            VariableData::I64(x) => x.sum(axis, keepdims).into(),
            VariableData::USIZE(x) => x.sum(axis, keepdims).into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "sum".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn sum_to(&self, shape: &[usize]) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.sum_to(shape)?.into(),
            VariableData::F64(x) => x.sum_to(shape)?.into(),
            VariableData::I32(x) => x.sum_to(shape)?.into(),
            VariableData::I64(x) => x.sum_to(shape)?.into(),
            VariableData::USIZE(x) => x.sum_to(shape)?.into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "sum_to".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.reshape(shape)?.into(),
            VariableData::F64(x) => x.reshape(shape)?.into(),
            VariableData::I32(x) => x.reshape(shape)?.into(),
            VariableData::I64(x) => x.reshape(shape)?.into(),
            VariableData::USIZE(x) => x.reshape(shape)?.into(),
            VariableData::Bool(x) => x.reshape(shape)?.into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "reshape".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn transpose(&self) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.transpose().into(),
            VariableData::F64(x) => x.transpose().into(),
            VariableData::I32(x) => x.transpose().into(),
            VariableData::I64(x) => x.transpose().into(),
            VariableData::USIZE(x) => x.transpose().into(),
            VariableData::Bool(x) => x.transpose().into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "transpose".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<VariableData> {
        Ok(match self {
            VariableData::F32(x) => x.broadcast_to(shape)?.into(),
            VariableData::F64(x) => x.broadcast_to(shape)?.into(),
            VariableData::I32(x) => x.broadcast_to(shape)?.into(),
            VariableData::I64(x) => x.broadcast_to(shape)?.into(),
            VariableData::USIZE(x) => x.broadcast_to(shape)?.into(),
            VariableData::Bool(x) => x.broadcast_to(shape)?.into(),
            _ => return Err(KDeZeroError::NotImplementedType(
                "broadcast_to".to_string(),
                self.data_type().to_string(),
            ).into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::Tensor;
    use ktensor::error::TensorError;

    #[test]
    fn add_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = VariableData::from(3.0f32);
        let z = x.add(&y)?;
        assert_eq!(z, VariableData::from(5.0f32));
        Ok(())
    }

    #[test]
    fn error_add_bool() -> Result<()> {
        let x = VariableData::from(true);
        let y = VariableData::from(false);
        match x.add(&y) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "add".to_string(),
                    format!("{:?}, {:?}", x.data_type(), y.data_type()),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn sub_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = VariableData::from(1.0f32);
        let z = x.sub(&y)?;
        assert_eq!(z, VariableData::from(1.0f32));
        Ok(())
    }

    #[test]
    fn error_sub_bool() -> Result<()> {
        let x = VariableData::from(true);
        let y = VariableData::from(false);
        match x.sub(&y) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "sub".to_string(),
                    format!("{:?}, {:?}", x.data_type(), y.data_type()),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn mul_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = VariableData::from(3.0f32);
        let z = x.mul(&y)?;
        assert_eq!(z, VariableData::from(6.0f32));
        Ok(())
    }

    #[test]
    fn error_mul_bool() -> Result<()> {
        let x = VariableData::from(true);
        let y = VariableData::from(false);
        match x.mul(&y) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "mul".to_string(),
                    format!("{:?}, {:?}", x.data_type(), y.data_type()),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn div_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = VariableData::from(3.0f32);
        let z = x.div(&y)?;
        assert_eq!(z, VariableData::from(2.0f32 / 3.0f32));
        Ok(())
    }

    #[test]
    fn error_div_bool() -> Result<()> {
        let x = VariableData::from(true);
        let y = VariableData::from(false);
        match x.div(&y) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "div".to_string(),
                    format!("{:?}, {:?}", x.data_type(), y.data_type()),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn neg_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.neg()?;
        assert_eq!(y, VariableData::from(-2.0f32));
        Ok(())
    }

    #[test]
    fn error_neg_bool() -> Result<()> {
        let x = VariableData::from(true);
        match x.neg() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "neg".to_string(),
                    x.data_type().to_string(),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn scalar_add_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.scalar_add(3.0)?;
        assert_eq!(y, VariableData::from(5.0f32));
        Ok(())
    }

    #[test]
    fn error_scalar_add_bool() -> Result<()> {
        let x = VariableData::from(true);
        match x.scalar_add(3.0) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "scalar_add".to_string(),
                    x.data_type().to_string(),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn scalar_mul_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.scalar_mul(3.0)?;
        assert_eq!(y, VariableData::from(6.0f32));
        Ok(())
    }

    #[test]
    fn error_scalar_mul_bool() -> Result<()> {
        let x = VariableData::from(true);
        match x.scalar_mul(3.0) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "scalar_mul".to_string(),
                    x.data_type().to_string(),
                ));
            }
        }
        Ok(())
    }

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

    #[test]
    fn exp_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.exp()?;
        assert_eq!(y, VariableData::from(2.0f32.exp()));
        Ok(())
    }

    #[test]
    fn error_exp_bool() -> Result<()> {
        let x = VariableData::from(true);
        match x.exp() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::NotImplementedType(
                    "exp".to_string(),
                    x.data_type().to_string(),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn pow_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.pow(3.0)?;
        assert_eq!(y, VariableData::from(8.0f32));
        Ok(())
    }

    #[test]
    fn pow_i32() -> Result<()> {
        let x = VariableData::from(2i32);
        let y = x.pow(3.0)?;
        assert_eq!(y, VariableData::from(8i32));
        Ok(())
    }

    #[test]
    fn error_pow_bool() -> Result<()> {
        let x = VariableData::from(true);
        match x.pow(3.0) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e,
                    KDeZeroError::NotImplementedType(
                        "pow".to_string(),
                        x.data_type().to_string(),
                    )
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sin_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.sin()?;
        assert_eq!(y, VariableData::from(2.0f32.sin()));
        Ok(())
    }

    #[test]
    fn error_sin_i32() -> Result<()> {
        let x = VariableData::from(2i32);
        match x.sin() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(
                    e,
                    KDeZeroError::NotImplementedType(
                        "sin".to_string(),
                        x.data_type().to_string(),
                    )
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cos_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.cos()?;
        assert_eq!(y, VariableData::from(2.0f32.cos()));
        Ok(())
    }

    #[test]
    fn error_cos_i32() -> Result<()> {
        let x = VariableData::from(2i32);
        match x.cos() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(
                    e,
                    KDeZeroError::NotImplementedType(
                        "cos".to_string(),
                        x.data_type().to_string(),
                    )
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tanh_f32() -> Result<()> {
        let x = VariableData::from(2.0f32);
        let y = x.tanh()?;
        assert_eq!(y, VariableData::from(2.0f32.tanh()));
        Ok(())
    }

    #[test]
    fn error_tanh_i32() -> Result<()> {
        let x = VariableData::from(2i32);
        match x.tanh() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(
                    e,
                    KDeZeroError::NotImplementedType(
                        "tanh".to_string(),
                        x.data_type().to_string(),
                    )
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sum_to_f32() -> Result<()> {
        let x = VariableData::from(Tensor::<f64>::arrange([2, 3])?);
        let y = x.sum_to(&[3])?;
        assert_eq!(y, Tensor::<f64>::arrange([2, 3])?.sum_to([3])?.into());
        Ok(())
    }

    #[test]
    fn error_sum_to_mismatch() -> Result<()> {
        let x = VariableData::from(Tensor::<f64>::arrange([1, 6])?);
        match x.sum_to(&[2, 2]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>()?;
                assert_eq!(
                    e,
                    TensorError::ShapeError(
                        vec![1, 6], vec![2, 2]
                    )
                );
            }
        }
        Ok(())
    }

    #[test]
    fn error_sum_to_bool() -> Result<()> {
        let x = VariableData::from(Tensor::full(true, [2, 3]));
        match x.sum_to(&[3]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(
                    e,
                    KDeZeroError::NotImplementedType(
                        "sum_to".to_string(),
                        x.data_type().to_string(),
                    )
                );
            }
        }
        Ok(())
    }

    #[test]
    fn reshape_f32() -> Result<()> {
        let x = VariableData::from(Tensor::<f64>::arrange([1, 6])?);
        let y = x.reshape(&[2, 3])?;
        assert_eq!(y, Tensor::<f64>::arrange([2, 3])?.into());
        Ok(())
    }

    #[test]
    fn reshape_no_match() -> Result<()> {
        let x = VariableData::from(Tensor::<f64>::arrange([1, 6])?);
        match x.reshape(&[2, 2]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>()?;
                assert_eq!(
                    e,
                    TensorError::ShapeSizeError(
                        6, 4
                    )
                );
            }
        }
        Ok(())
    }

    #[test]
    fn transpose_f32() -> Result<()> {
        let x = VariableData::from(Tensor::<f64>::arrange([1, 6])?);
        let y = x.transpose()?;
        assert_eq!(y, Tensor::<f64>::arrange([6, 1])?.into());
        Ok(())
    }

    #[test]
    fn broadcast_to_f32() -> Result<()> {
        let x = VariableData::from(Tensor::<f64>::arrange([3,])?);
        let y = x.broadcast_to(&[2, 3])?;
        assert_eq!(y, Tensor::<f64>::arrange([3,])?.broadcast_to([2, 3])?.into());
        Ok(())
    }

    #[test]
    fn error_broadcast_to_mismatch() -> Result<()> {
        let x = VariableData::from(Tensor::<f64>::arrange([1, 6])?);
        match x.broadcast_to(&[2, 2]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>()?;
                assert_eq!(
                    e,
                    TensorError::ShapeError(
                        vec![1, 6], vec![2, 2]
                    )
                );
            }
        }
        Ok(())
    }
}
