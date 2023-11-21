use anyhow::Result;
use crate::Variable;
use super::transpose;
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct MatMul {}

impl MatMul {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for MatMul {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        let x0 = xs[0].data();
        let x1 = xs[1].data();
        let y = x0.matmul(&x1)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        check_variable_count(&gys, 1)?;
        let x0 = xs[0];
        let x1 = xs[1];
        let gy = gys[0];
        let gx = matmul(&gy, &transpose(&x1)?)?;
        let gw = matmul(&transpose(&x0)?, &gy)?;
        Ok(vec![gx, gw])
    }

    fn name(&self) -> String {
        "MatMul".to_string()
    }
}

pub fn matmul(x0: &Variable, x1: &Variable) -> Result<Variable> {
    let mut func = Function::new(MatMul::new());
    let mut ys = func.forward(&[x0.clone(), x1.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::Tensor;
    use crate::error::KDeZeroError;

    #[test]
    fn matmul_forward() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let y = MatMul::new().forward(vec![&x0, &x1])?;
        assert_eq!(*y[0].data(), Tensor::<f64>::new(vec![5.0, 14.0], [2, 1])?.into());
        Ok(())
    }

    #[test]
    fn error_matmul_forward_invalid_variable_count() -> Result<()> {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        match MatMul::new().forward(vec![&x0, &x1, &x0]) {
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
    fn matmul_backward() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let dy = Variable::new(Tensor::<f64>::arrange([2, 1])?.into());
        let f = MatMul::new();
        let dx = f.backward(vec![&x0, &x1], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), dy.data().matmul(&x1.data().transpose()?)?);
        assert_eq!(*dx[1].data(), x0.data().transpose()?.matmul(&dy.data())?);
        Ok(())
    }

    #[test]
    fn matmul_normal() -> Result<()> {
        let x0 = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let x1 = Variable::new(Tensor::<f64>::arrange([3, 1])?.into());
        let y = matmul(&x0, &x1)?;
        assert_eq!(*y.data(), Tensor::<f64>::new(vec![5.0, 14.0], [2, 1])?.into());
        Ok(())
    }
}
