use anyhow::Result;
use crate::Variable;
use super::mul;
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Relu {}

impl Relu {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for Relu {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let zero = x.zeros_like()?;
        let y = x.maximum(&zero)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let mask = x.data().greater_zero()?;
        let gx = mul(gy, &mask.into())?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Relu".to_string()
    }
}

pub fn relu(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Relu::new());
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
    fn relu_forward() -> Result<()> {
        let x = Variable::new(
            Tensor::<f64>::new(vec![-1.0, 0.0, 0.5], [3])?.into()
        );
        let y = Relu::new().forward(vec![&x])?;
        assert_eq!(
            *y[0].data(),
            Tensor::<f64>::new(vec![0.0, 0.0, 0.5], [3])?.into());
        Ok(())
    }

    #[test]
    fn error_relu_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(0.0);
        match Relu::new().forward(vec![&x.clone(), &x]) {
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
    fn relu_backward() -> Result<()> {
        let x = Variable::new(
            Tensor::<f64>::new(vec![-1.0, 0.0, 0.5], [3])?.into()
        );
        let dy = Variable::new(
            Tensor::<f64>::new(vec![2.0, 1.0, -1.0], [3])?.into()
        );
        let f = Relu::new();
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(
            *dx[0].data(),
            Tensor::<f64>::new(vec![0.0, 0.0, -1.0], [3])?.into());
        Ok(())
    }

    #[test]
    fn error_relu_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::new(
            Tensor::<f64>::new(vec![-1.0, 0.0, 0.5], [3])?.into()
        );
        let dy = Variable::new(
            Tensor::<f64>::new(vec![2.0, 1.0, -1.0], [3])?.into()
        );
        let f = Relu::new();
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
    fn error_relu_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::new(
            Tensor::<f64>::new(vec![-1.0, 0.0, 0.5], [3])?.into()
        );
        let dy = Variable::new(
            Tensor::<f64>::new(vec![2.0, 1.0, -1.0], [3])?.into()
        );
        let f = Relu::new();
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
    fn relu_normal() -> Result<()> {
        let x = Variable::new(
            Tensor::<f64>::new(vec![-1.0, 0.0, 0.5], [3])?.into()
        );
        let y = relu(&x)?;
        assert_eq!(
            *y.data(),
            Tensor::<f64>::new(vec![0.0, 0.0, 0.5], [3])?.into());
        Ok(())
    }
}
