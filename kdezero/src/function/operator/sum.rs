use anyhow::Result;
use crate::Variable;
use super::{broadcast_to, reshape};
use super::super::{FunctionContent, Function};
use super::super::function_helper::check_variable_count;

#[derive(Debug)]
pub struct Sum {
    pub axis: Option<Vec<usize>>,
    pub keepdims: bool,
}

impl Sum {
    pub fn new(axis: Option<Vec<usize>>, keepdims: bool) -> Self {
        Self { axis, keepdims }
    }
}

impl FunctionContent for Sum {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = x.sum(self.axis.as_ref(), self.keepdims)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let input_shape = x.shape();
        
        let gx = if !self.keepdims && input_shape.len() != 0 {
            let mut axis = match self.axis.as_ref() {
                Some(axis) => axis.clone(),
                None => (0..input_shape.len()).collect(),
            };
            axis.sort();
            let mut output_shape = gy.shape().to_vec();
            for i in axis.iter() {
                output_shape.insert(*i, 1);
            }
            let gx = reshape(gy, &output_shape)?;
            broadcast_to(&gx, &input_shape)?
        } else {
            broadcast_to(gy, &input_shape)?
        };

        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Sum".to_string()
    }
}

pub fn sum(x: &Variable, axis: Option<Vec<usize>>, keepdims: bool) -> Result<Variable> {
    let mut func = Function::new(Sum::new(axis, keepdims));
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

pub fn sum_keepdims(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Sum::new(None, true));
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

pub fn sum_axis(x: &Variable, axis: Vec<usize>, keepdims: bool) -> Result<Variable> {
    let mut func = Function::new(Sum::new(Some(axis), keepdims));
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

pub fn sum_all(x: &Variable) -> Result<Variable> {
    let mut func = Function::new(Sum::new(None, false));
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
    fn sum_forward() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let y = Sum::new(None, false).forward(vec![&x])?;
        assert_eq!(*y[0].data(), Tensor::<f64>::arrange([2, 3])?.sum(Some([0, 1]), false).into());
        Ok(())
    }

    #[test]
    fn error_sum_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match Sum::new(None, false).forward(vec![&x.clone(), &x]) {
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
    fn sum_backward() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let dy = Variable::new(Tensor::full(3.0, []).into());
        let f = Sum::new(None, false);
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), x.data().full_like(3.0)?);
        Ok(())
    }

    #[test]
    fn sum_backward_reshape() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let dy = Variable::new(Tensor::full(3.0, [1]).into());
        let f = Sum::new(Some(vec![1]), false);
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), x.data().full_like(3.0)?);
        Ok(())
    }

    #[test]
    fn error_sum_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Sum::new(None, false);
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
    fn error_sum_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::from(2.0);
        let dy = Variable::from(3.0);
        let f = Sum::new(None, false);
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
    fn sum_normal() -> Result<()> {
        let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
        let y = sum(&x, None, false)?;
        assert_eq!(*y.data(), Tensor::<f64>::arrange([2, 3])?.sum(Some([0, 1]), false).into());
        Ok(())
    }
}
