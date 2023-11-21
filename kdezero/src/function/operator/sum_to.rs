use anyhow::Result;
use crate::Variable;
use super::broadcast_to;
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct SumTo {
    pub shape: Vec<usize>,
}

impl SumTo {
    pub fn new(shape: &[usize]) -> Self {
        Self { shape: shape.to_vec() }
    }
}

impl FunctionContent for SumTo {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.sum_to(&self.shape)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let shape = x.shape();
        let gx = broadcast_to(gy, &shape)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "SumTo".to_string()
    }
}

pub fn sum_to(x: &Variable, shape: &[usize]) -> Result<Variable> {
    let mut func = Function::new(SumTo::new(shape));
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktensor::Tensor;
    use ktensor::error::TensorError;
    use crate::error::KDeZeroError;

    #[test]
    fn sum_to_forward() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let y = SumTo::new(&[3]).forward(vec![&x])?;
        assert_eq!(*y[0].data(), Tensor::<f64>::arrange([2, 3])?.sum_to([3])?.into());
        Ok(())
    }

    #[test]
    fn error_sum_to_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match SumTo::new(&[2, 3]).forward(vec![&x.clone(), &x]) {
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
    fn error_sum_to_mismatch() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([1, 6])?.into());
        match SumTo::new(&[2, 2]).forward(vec![&x]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>()?;
                assert_eq!(e, TensorError::ShapeError(
                    vec![1, 6], vec![2, 2]
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn sum_to_backward() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let dy = Variable::new(Tensor::full(3.0, [3]).into());
        let f = SumTo::new(&[3]);
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), x.data().full_like(3.0)?);
        Ok(())
    }

    #[test]
    fn error_sum_to_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = SumTo::new(&[2, 3]);
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
    fn error_sum_to_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = SumTo::new(&[2, 3]);
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
    fn sum_to_normal() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let y = sum_to(&x, &[3])?;
        assert_eq!(*y.data(), Tensor::<f64>::arrange([2, 3])?.sum_to([3])?.into());
        Ok(())
    }
}
