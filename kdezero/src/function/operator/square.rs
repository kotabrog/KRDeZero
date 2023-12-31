use anyhow::Result;
use crate::Variable;
use super::mul;
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Square {}

impl Square {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Square {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.square()?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let gx = mul(gy,
            &mul(x, &x.data().full_like(2.0)?.into())?)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Square".to_string()
    }
}

pub fn square(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Square::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn square_forward() -> Result<()> {
        let x = Variable::from(2.0);
        let y = Square::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), 4.0.into());
        Ok(())
    }

    #[test]
    fn error_square_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match Square::new().forward(vec![&x.clone(), &x]) {
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
    fn square_backward() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Square::new();
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), 12.0.into());
        Ok(())
    }

    #[test]
    fn error_square_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Square::new();
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
    fn error_square_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Square::new();
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
    fn square_normal() -> Result<()> {
        let x = Variable::from(2.0);
        let y = square(&x)?;
        assert_eq!(*y.data(), 4.0.into());
        Ok(())
    }
}
