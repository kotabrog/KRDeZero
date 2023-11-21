use anyhow::Result;
use crate::Variable;
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Mul {}

impl Mul {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Mul {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        let x0 = xs[0].data();
        let x1 = xs[1].data();
        let y = x0.mul(&x1)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        check_variable_count(&gys, 1)?;
        let x0 = xs[0];
        let x1 = xs[1];
        let gy = gys[0];
        let gx0 = mul(gy, x1)?;
        let gx1 = mul(gy, x0)?;
        Ok(vec![gx0, gx1])
    }

    fn name(&self) -> String {
        "Mul".to_string()
    }
}

pub fn mul(x0: &Variable, x1: &Variable) -> Result<Variable> {
    let mut func = Function::new(Mul::new());
    let mut ys = func.forward(&[x0.clone(), x1.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn mul_forward() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = Mul::new().forward(vec![&x0, &x1])?;
        assert_eq!(*y[0].data(), 6.0.into());
        Ok(())
    }

    #[test]
    fn error_mul_forward_invalid_variable_count() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        match Mul::new().forward(vec![&x0, &x1, &x0]) {
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
    fn mul_backward() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let dy = Variable::from(4.0);
        let f = Mul::new();
        let dx = f.backward(vec![&x0, &x1], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), 12.0.into());
        assert_eq!(*dx[1].data(), 8.0.into());
        Ok(())
    }

    #[test]
    fn mul_normal() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = mul(&x0, &x1)?;
        assert_eq!(*y.data(), 6.0.into());
        Ok(())
    }
}
