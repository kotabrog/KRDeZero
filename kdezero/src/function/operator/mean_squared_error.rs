use anyhow::Result;
use crate::Variable;
use super::{sub, broadcast_to, mul, neg};
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct MeanSquaredError {}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for MeanSquaredError {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        let x0 = xs[0].data();
        let x1 = xs[1].data();
        let diff = x0.sub(&x1)?;
        let n = diff.len() as f64;
        let y = diff.pow(2.0)?
            .sum(None, false)?.scalar_mul(1.0 / n)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        check_variable_count(&gys, 1)?;
        let x0 = xs[0];
        let x1 = xs[1];
        let diff = sub(x0, x1)?;
        let gy = gys[0];
        let gy = broadcast_to(gy, &*diff.shape())?;
        let n = diff.len() as f64;
        let c = broadcast_to(&(2.0 / n).into(), &*diff.shape())?;
        let gx0 = mul(&mul(&gy, &diff)?, &c)?;
        let gx1 = neg(&gx0)?;
        Ok(vec![gx0, gx1])
    }

    fn name(&self) -> String {
        "MeanSquaredError".to_string()
    }
}

pub fn mean_squared_error(x0: &Variable, x1: &Variable) -> Result<Variable> {
    let mut func = Function::new(MeanSquaredError::new());
    let mut ys = func.forward(&[x0.clone(), x1.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::Tensor;
    use crate::error::KDeZeroError;

    #[test]
    fn mean_squared_error_forward() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::full(3.0, [2, 3]).into());
        let y = MeanSquaredError::new().forward(vec![&x0, &x1])?;
        assert_eq!(*y[0].data(), 9.5.into());
        Ok(())
    }

    #[test]
    fn error_mean_squared_error_forward_invalid_variable_count() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        match MeanSquaredError::new().forward(vec![&x0, &x1, &x0]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    2, 3));
                }
            }
        Ok(())
    }

    #[test]
    fn mean_squared_error_backward() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::full(3.0, [2, 3]).into());
        let dy = Variable::from(4.0);
        let f = MeanSquaredError::new();
        let dx = f.backward(vec![&x0, &x1], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(),
            Tensor::new([-12.0, -8.0, -4.0, 0.0, 4.0, 8.0], vec![2, 3])?.into());
        assert_eq!(*dx[1].data(),
            Tensor::new([12.0, 8.0, 4.0, 0.0, -4.0, -8.0], vec![2, 3])?.into());
        Ok(())
    }

    #[test]
    fn mean_squared_error_normal() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::full(3.0, [2, 3]).into());
        let y = mean_squared_error(&x0, &x1)?;
        assert_eq!(*y.data(), 9.5.into());
        Ok(())
    }
}
