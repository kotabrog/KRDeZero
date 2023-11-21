use anyhow::Result;
use crate::Variable;
use super::{transpose, matmul};
use super::super::{FunctionContent, Function};
use crate::utility::{check_variable_count, check_variable_count_between};

#[derive(Debug)]
pub struct Linear {}

impl Linear {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Linear {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        let len = check_variable_count_between(&xs, 2, 4)?;
        let x0 = xs[0].data();
        let x1 = xs[1].data();
        let y = x0.matmul(&x1)?;
        if len == 3 {
            let x2 = xs[2].data();
            let y = y.add(&x2)?;
            Ok(vec![y.into()])
        } else {
            Ok(vec![y.into()])
        }
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        let len = check_variable_count_between(&xs, 2, 4)?;
        check_variable_count(&gys, 1)?;
        let x0 = xs[0];
        let x1 = xs[1];
        let gy = gys[0];
        let gx0 = matmul(&gy, &transpose(&x1)?)?;
        let gx1 = matmul(&transpose(&x0)?, &gy)?;
        if len == 3 {
            Ok(vec![gx0, gx1, gy.clone()])
        } else {
            Ok(vec![gx0, gx1])
        }
    }

    fn name(&self) -> String {
        "Linear".to_string()
    }
}

pub fn linear(x0: &Variable, x1: &Variable, x2: Option<&Variable>) -> Result<Variable> {
    let mut func = Function::new(Linear::new());
    let mut ys = if let Some(x2) = x2 {
        func.forward(&[x0.clone(), x1.clone(), x2.clone()])?
    } else {
        func.forward(&[x0.clone(), x1.clone()])?
    };
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::Tensor;
    use crate::error::KDeZeroError;

    #[test]
    fn linear_forward() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let y = Linear::new().forward(vec![&x0, &x1])?;
        assert_eq!(*y[0].data(), Tensor::<f64>::new(vec![5.0, 14.0], [2, 1])?.into());
        Ok(())
    }

    #[test]
    fn linear_forward_with_b() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let x2 = Variable::new(Tensor::<f64>::arrange([2, 1])?.into());
        let y = Linear::new().forward(vec![&x0, &x1, &x2])?;
        assert_eq!(*y[0].data(), Tensor::<f64>::new(vec![5.0, 15.0], [2, 1])?.into());
        Ok(())
    }

    #[test]
    fn error_linear_forward_invalid_variable_count() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        match Linear::new().forward(vec![&x0, &x1, &x0, &x1]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::OutOfRangeVariableCount(
                    4, 2, 4));
                }
            }
        Ok(())
    }

    #[test]
    fn linear_backward() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let dy = Variable::new(Tensor::<f64>::arrange([2, 1])?.into());
        let f = Linear::new();
        let dx = f.backward(vec![&x0, &x1], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), dy.data().matmul(&x1.data().transpose()?)?);
        assert_eq!(*dx[1].data(), x0.data().transpose()?.matmul(&dy.data())?);
        Ok(())
    }

    #[test]
    fn linear_backward_with_b() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let x2 = Variable::new(Tensor::<f64>::arrange([2, 1])?.into());
        let dy = Variable::new(Tensor::<f64>::arrange([2, 1])?.into());
        let f = Linear::new();
        let dx = f.backward(vec![&x0, &x1, &x2], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), dy.data().matmul(&x1.data().transpose()?)?);
        assert_eq!(*dx[1].data(), x0.data().transpose()?.matmul(&dy.data())?);
        assert_eq!(*dx[2].data(), *dy.data());
        Ok(())
    }

    #[test]
    fn error_linear_backward_invalid_variable_count_dy() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let dy = Variable::from(4.0);
        let f = Linear::new();
        match f.backward(vec![&x0, &x1], vec![], vec![&dy, &dy]) {
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
    fn linear_normal() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let y = linear(&x0, &x1, None)?;
        assert_eq!(*y.data(), Tensor::<f64>::new(vec![5.0, 14.0], [2, 1])?.into());
        Ok(())
    }

    #[test]
    fn linear_with_b() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let x2 = Variable::new(Tensor::<f64>::arrange([2, 1])?.into());
        let y = linear(&x0, &x1, Some(&x2))?;
        assert_eq!(*y.data(), Tensor::<f64>::new(vec![5.0, 15.0], [2, 1])?.into());
        Ok(())
    }
}
