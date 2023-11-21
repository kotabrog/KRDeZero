use anyhow::Result;
use crate::Variable;
use super::{mul, sub, square};
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Tanh {}

impl Tanh {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Tanh {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.tanh()?;
        Ok(vec![y.into()])
    }

    fn backward(&self, _xs: Vec<&Variable>, ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&ys, 1)?;
        check_variable_count(&gys, 1)?;
        let y = ys[0];
        let gy = gys[0];
        let gx = mul(
            gy,
            &sub(&y.data().full_like(1.0)?.into(), &square(y)?)?
        )?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Tanh".to_string()
    }
}

pub fn tanh(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Tanh::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn tanh_forward() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let y = Tanh::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), std::f64::consts::FRAC_PI_4.tanh().into());
        Ok(())
    }

    #[test]
    fn error_tanh_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        match Tanh::new().forward(vec![&x.clone(), &x]) {
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
    fn tanh_backward() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Tanh::new();
        let y = f.forward(vec![&x])?;
        let dx = f.backward(vec![&x], vec![&y[0]], vec![&dy])?;
        assert_eq!(*dx[0].data(), (3.0 * (1.0 - y[0].data().to_f64_tensor()?.to_scalar()?.powi(2))).into());
        Ok(())
    }

    #[test]
    fn error_tanh_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Tanh::new();
        let y = f.forward(vec![&x])?;
        match f.backward(vec![&x], vec![&y[0]], vec![&dy, &dy]) {
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
    fn error_tanh_backward_invalid_variable_count_y() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Tanh::new();
        let y = f.forward(vec![&x])?;
        match f.backward(vec![&x], vec![&y[0], &y[0]], vec![&dy]) {
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
    fn tanh_normal() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let y = tanh(&x)?;
        assert_eq!(*y.data(), std::f64::consts::FRAC_PI_4.tanh().into());
        Ok(())
    }
}
