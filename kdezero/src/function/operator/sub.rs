use anyhow::Result;
use crate::Variable;
use super::neg;
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Sub {}

impl Sub {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Sub {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        let x0 = xs[0].data();
        let x1 = xs[1].data();
        let y = x0.sub(&x1)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        check_variable_count(&gys, 1)?;
        let gy = gys[0];
        let gx0 = gy.clone();
        let gx1 = neg(gy)?;
        Ok(vec![gx0, gx1])
    }

    fn name(&self) -> String {
        "Sub".to_string()
    }
}

pub fn sub(x0: &Variable, x1: &Variable) -> Result<Variable> {
    let mut func = Function::new(Sub::new());
    let mut ys = func.forward(&[x0.clone(), x1.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn sub_forward() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = Sub::new().forward(vec![&x0, &x1])?;
        assert_eq!(*y[0].data(), (-1.0).into());
        Ok(())
    }

    #[test]
    fn error_sub_forward_invalid_variable_count() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        match Sub::new().forward(vec![&x0, &x1, &x0]) {
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
    fn sub_backward() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let dy = Variable::from(4.0);
        let f = Sub::new();
        let dx = f.backward(vec![&x0, &x1], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), 4.0.into());
        assert_eq!(*dx[1].data(), (-4.0).into());
        Ok(())
    }

    #[test]
    fn sub_normal() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = sub(&x0, &x1)?;
        assert_eq!(*y.data(), (-1.0).into());
        Ok(())
    }
}
