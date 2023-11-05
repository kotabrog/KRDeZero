use anyhow::Result;
use crate::Variable;
use super::super::{FunctionContent, Function};
use super::super::function_helper::check_variable_count;

#[derive(Debug)]
pub struct Sin {}

impl Sin {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Sin {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.sin()?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0].data();
        let gy = gys[0].data();
        let gx = gy.mul(&x.cos()?)?;
        Ok(vec![gx.into()])
    }

    fn name(&self) -> String {
        "Sin".to_string()
    }
}

pub fn sin(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Sin::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn sin_forward() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let y = Sin::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), std::f64::consts::FRAC_PI_4.sin().into());
        Ok(())
    }

    #[test]
    fn error_sin_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        match Sin::new().forward(vec![&x.clone(), &x]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    1,
                    2,
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn sin_backward() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Sin::new();
        let dx = f.backward(vec![&x], vec![&dy])?;
        assert_eq!(*dx[0].data(), (3.0 * std::f64::consts::FRAC_PI_4.cos()).into());
        Ok(())
    }

    #[test]
    fn error_sin_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Sin::new();
        match f.backward(vec![&x], vec![&dy, &dy]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    1,
                    2,
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn error_sin_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Sin::new();
        match f.backward(vec![&x, &x], vec![&dy]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    1,
                    2,
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn sin_normal() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let y = sin(&x)?;
        assert_eq!(*y.data(), std::f64::consts::FRAC_PI_4.sin().into());
        Ok(())
    }
}
