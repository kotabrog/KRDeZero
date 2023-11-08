use anyhow::Result;
use crate::Variable;
use super::super::{FunctionContent, Function};
use super::super::function_helper::check_variable_count;

#[derive(Debug)]
pub struct Transpose {}

impl Transpose {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Transpose {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.transpose()?;
        Ok(vec![y.into()])
    }

    fn backward(&self, _xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&gys, 1)?;
        let gy = gys[0];
        let gx = transpose(gy)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Transpose".to_string()
    }
}

pub fn transpose(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Transpose::new());
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::Tensor;
    use crate::error::KDeZeroError;

    #[test]
    fn transpose_forward() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([3, 2])?.into());
        let y = Transpose::new().forward(vec![&x])?;
        assert_eq!(*y[0].data(), Tensor::<f64>::arrange([3, 2])?.transpose().into());
        Ok(())
    }

    #[test]
    fn error_transpose_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match Transpose::new().forward(vec![&x.clone(), &x]) {
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
    fn transpose_backward() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([3, 2])?.into());
        let dy = Variable::new(Tensor::full(3.0, [2, 3]).into());
        let f = Transpose::new();
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), x.data().full_like(3.0)?);
        Ok(())
    }

    #[test]
    fn error_transpose_backward_invalid_variable_count_dy() -> Result<()> {
        let dy = Variable::from(3.0);
        let f = Transpose::new();
        match f.backward(vec![], vec![], vec![&dy, &dy]) {
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
    fn transpose_normal() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([3, 2])?.into());
        let y = transpose(&x)?;
        assert_eq!(*y.data(), Tensor::<f64>::arrange([3, 2])?.transpose().into());
        Ok(())
    }
}
