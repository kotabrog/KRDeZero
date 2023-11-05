use anyhow::Result;
use crate::Variable;
use super::mul;
use super::super::{FunctionContent, Function};
use super::super::function_helper::check_variable_count;

#[derive(Debug)]
pub struct Pow {
    pub c: f64,
}

impl Pow {
    pub fn new(c: f64) -> Self {
        Self { c }
    }
}

impl FunctionContent for Pow {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.pow(self.c)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let gx = mul(
            &mul(&pow(x, self.c - 1.0)?, gy)?,
            &x.data().full_like(self.c)?.into())?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Pow".to_string()
    }
}

pub fn pow(x: &Variable, c: f64) -> Result<Variable> {
    let mut func = Function::new(Pow::new(c));
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn pow_forward() -> Result<()> {
        let x = Variable::from(2.0);
        let y = Pow::new(3.0).forward(vec![&x])?;
        assert_eq!(*y[0].data(), 8.0.into());
        Ok(())
    }

    #[test]
    fn error_pow_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match Pow::new(3.0).forward(vec![&x.clone(), &x]) {
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
    fn pow_backward() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Pow::new(3.0);
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), (36.0).into());
        Ok(())
    }

    #[test]
    fn error_pow_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Pow::new(3.0);
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
    fn error_pow_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Pow::new(3.0);
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
    fn pow_normal() -> Result<()> {
        let x = Variable::from(2.0);
        let y = pow(&x, 3.0)?;
        assert_eq!(*y.data(), 8.0.into());
        Ok(())
    }
}
