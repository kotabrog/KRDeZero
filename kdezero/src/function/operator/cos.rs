use anyhow::Result;
use crate::Variable;
use super::{mul, sin, neg};
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Cos {}

impl Cos {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Cos {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.cos()?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let gx = neg(&mul(&gy, &sin(x)?)?)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Cos".to_string()
    }
}

pub fn cos(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Cos::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn cos_forward() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let y = Cos::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), std::f64::consts::FRAC_PI_4.cos().into());
        Ok(())
    }

    #[test]
    fn error_cos_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        match Cos::new().forward(vec![&x.clone(), &x]) {
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
    fn cos_backward() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Cos::new();
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), (-3.0 * std::f64::consts::FRAC_PI_4.sin()).into());
        Ok(())
    }

    #[test]
    fn error_cos_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Cos::new();
        match f.backward(vec![&x], vec![], vec![&dy, &dy]) {
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
    fn error_cos_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let dy = Variable::from(3.0);
        let f = Cos::new();
        match f.backward(vec![&x, &x], vec![], vec![&dy]) {
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
    fn cos_normal() -> Result<()> {
        let x = Variable::from(std::f64::consts::FRAC_PI_4);
        let y = cos(&x)?;
        assert_eq!(*y.data(), std::f64::consts::FRAC_PI_4.cos().into());
        Ok(())
    }
}
