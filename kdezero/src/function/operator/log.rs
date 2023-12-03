use anyhow::Result;
use crate::Variable;
use super::div;
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Log {}

impl Log {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Log {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.log()?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let gx = div(gy, x)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Log".to_string()
    }
}

pub fn log(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Log::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn log_forward() -> Result<()> {
        let x = Variable::from(2.0);
        let y = Log::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), 2.0f64.ln().into());
        Ok(())
    }

    #[test]
    fn error_log_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match Log::new().forward(vec![&x.clone(), &x]) {
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
    fn log_backward() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Log::new();
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), (3.0 / 2.0).into());
        Ok(())
    }

    #[test]
    fn error_log_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Log::new();
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
    fn error_log_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Log::new();
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
    fn log_normal() -> Result<()> {
        let x = Variable::from(2.0);
        let y = log(&x)?;
        assert_eq!(*y.data(), 2.0f64.ln().into());
        Ok(())
    }
}
