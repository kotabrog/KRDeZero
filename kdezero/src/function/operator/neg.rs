use anyhow::Result;
use crate::Variable;
use super::super::{FunctionContent, Function};
use super::super::function_helper::check_variable_count;

#[derive(Debug)]
pub struct Neg {}

impl Neg {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Neg {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.neg()?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let gy = gys[0];
        let gx = neg(gy)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Neg".to_string()
    }
}

pub fn neg(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Neg::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn neg_forward() -> Result<()> {
        let x = Variable::from(2.0);
        let y = Neg::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), (-2.0).into());
        Ok(())
    }

    #[test]
    fn error_neg_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match Neg::new().forward(vec![&x, &x]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    1, 2));
                }
            }
        Ok(())
    }

    #[test]
    fn neg_backward() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(4.0);
        let f = Neg::new();
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), (-4.0).into());
        Ok(())
    }

    #[test]
    fn neg_normal() -> Result<()> {
        let x = Variable::from(2.0);
        let y = neg(&x)?;
        assert_eq!(*y.data(), (-2.0).into());
        Ok(())
    }
}
