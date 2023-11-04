use anyhow::Result;
use crate::Variable;
use super::super::{FunctionContent, Function};
use super::super::function_helper::check_variable_count;

#[derive(Debug)]
pub struct Div {}

impl Div {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Div {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        let x0 = xs[0].data();
        let x1 = xs[1].data();
        let y = x0.div(&x1)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        check_variable_count(&gys, 1)?;
        let x0 = xs[0].data();
        let x1 = xs[1].data();
        let gy = gys[0].data();
        let gx0 = gy.div(&x1)?;
        let gx1 = gy.mul(&x0.neg()?.div(&x1.square()?)?)?;
        Ok(vec![gx0.into(), gx1.into()])
    }
}

pub fn div(x0: &Variable, x1: &Variable) -> Result<Variable> {
    let mut func = Function::new(Div::new());
    let mut ys = func.forward(&[x0.clone(), x1.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn div_forward() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = Div::new().forward(vec![&x0, &x1])?;
        assert_eq!(*y[0].data(), (2.0 / 3.0).into());
        Ok(())
    }

    #[test]
    fn error_div_forward_invalid_variable_count() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        match Div::new().forward(vec![&x0, &x1, &x0]) {
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
    fn div_backward() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let dy = Variable::from(4.0);
        let f = Div::new();
        let dx = f.backward(vec![&x0, &x1], vec![&dy])?;
        assert_eq!(*dx[0].data(), (4.0 / 3.0).into());
        assert_eq!(*dx[1].data(), (4.0 * ((-2.0) / (3.0 * 3.0))).into());
        Ok(())
    }

    #[test]
    fn div_normal() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = div(&x0, &x1)?;
        assert_eq!(*y.data(), (2.0 / 3.0).into());
        Ok(())
    }
}
