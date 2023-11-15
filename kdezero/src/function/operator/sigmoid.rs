use anyhow::Result;
use crate::Variable;
use super::{mul, sub};
use super::super::{FunctionContent, Function};
use super::super::function_helper::check_variable_count;

#[derive(Debug)]
pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Sigmoid {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.scalar_mul(0.5)?
            .tanh()?
            .scalar_add(1.0)?
            .scalar_mul(0.5)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, _xs: Vec<&Variable>, ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&ys, 1)?;
        check_variable_count(&gys, 1)?;
        let y = ys[0];
        let gy = gys[0];
        let gx = mul(
            gy,
            &mul(
                &sub(&y.data().full_like(1.0)?.into(), &y)?,
                &y,
            )?,
        )?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Sigmoid".to_string()
    }
}

pub fn sigmoid(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Sigmoid::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn sigmoid_forward() -> Result<()> {
        let x = Variable::from(0.0);
        let y = Sigmoid::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), 0.5.into());
        Ok(())
    }

    #[test]
    fn error_sigmoid_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(0.0);
        match Sigmoid::new().forward(vec![&x.clone(), &x]) {
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
    fn sigmoid_backward() -> Result<()> {
        let y = Variable::from(0.5);
        let dy = Variable::from(3.0);
        let f = Sigmoid::new();
        let dx = f.backward(vec![], vec![&y], vec![&dy])?;
        assert_eq!(*dx[0].data(), (3.0 * 0.5 * 0.5).into());
        Ok(())
    }

    #[test]
    fn error_sigmoid_backward_invalid_variable_count_dy() -> Result<()> {
        let y = Variable::from(0.5);
        let dy = Variable::from(3.0);
        let f = Sigmoid::new();
        match f.backward(vec![], vec![&y], vec![&dy, &dy]) {
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
    fn error_sigmoid_backward_invalid_variable_count_y() -> Result<()> {
        let y = Variable::from(0.5);
        let dy = Variable::from(3.0);
        let f = Sigmoid::new();
        match f.backward(vec![], vec![&y, &y], vec![&dy]) {
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
    fn sigmoid_normal() -> Result<()> {
        let x = Variable::from(0.0);
        let y = sigmoid(&x)?;
        assert_eq!(*y.data(), 0.5.into());
        Ok(())
    }
}
