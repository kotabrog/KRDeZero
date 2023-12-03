use anyhow::Result;
use crate::Variable;
use super::{sub, mul, sum_axis, broadcast_to};
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct Softmax {
    pub axis: usize,
}

impl Softmax {
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl FunctionContent for Softmax {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let x_max = x.max_with_axis(self.axis, true)?
            .broadcast_to(x.shape())?;
        let y = x.sub(&x_max)?
            .exp()?;
        let y_sum = y.sum(Some(&vec![self.axis]), true)?
            .broadcast_to(x.shape())?;
        let y = y.div(&y_sum)?;
        Ok(vec![y.into()])
    }

    fn backward(&self, _xs: Vec<&Variable>, ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&ys, 1)?;
        check_variable_count(&gys, 1)?;
        let y = ys[0];
        let gy = gys[0];
        let gx = mul(y, gy)?;
        let sum_gx = sum_axis(&gx, vec![self.axis], true)?;
        let sum_gx = broadcast_to(&sum_gx, &y.shape())?;
        let gx = sub(&gx, &mul(&y, &sum_gx)?)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "Softmax".to_string()
    }
}

pub fn softmax(x: &Variable, axis: usize) -> Result<Variable> {
    let mut func = Function::new(Softmax::new(axis));
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;
    use ktensor::Tensor;

    #[test]
    fn softmax_forward() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, -2.0, 2.0, 1.0, 0.0, 3.0, 3.0],
            vec![4, 2],
        )?.into());
        let y = Softmax::new(1).forward(vec![&x])?;
        let y_data = y[0].data().clone();
        assert_eq!(y_data.shape(), &vec![4, 2]);
        let y_data = y_data.to_f64_tensor()?.get_data();
        assert!(y_data[0] < y_data[1]);
        assert!(y_data[2] < y_data[3]);
        assert!(y_data[4] > y_data[5]);
        assert_eq!(y_data[6], y_data[7]);
        assert!((y_data[0] + y_data[1] - 1.).abs() < 0.0001);
        assert!((y_data[2] + y_data[3] - 1.).abs() < 0.0001);
        assert!((y_data[4] + y_data[5] - 1.).abs() < 0.0001);
        assert!((y_data[6] + y_data[7] - 1.).abs() < 0.0001);
        Ok(())
    }

    #[test]
    fn error_softmax_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(0.0);
        match Softmax::new(1).forward(vec![&x.clone(), &x]) {
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
    fn softmax_backward() -> Result<()> {
        let y = Variable::new(Tensor::new(
            vec![0.0, 1.0, -2.0, 2.0, 1.0, 0.0, 3.0, 3.0],
            vec![4, 2],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![4, 2],
        )?.into());
        let f = Softmax::new(1);
        let dx = f.backward(vec![], vec![&y], vec![&dy])?;
        assert_eq!(*dx[0].shape(), vec![4, 2]);
        Ok(())
    }

    #[test]
    fn error_softmax_backward_invalid_variable_count_dy() -> Result<()> {
        let y = Variable::from(0.5);
        let dy = Variable::from(3.0);
        let f = Softmax::new(1);
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
    fn error_softmax_backward_invalid_variable_count_y() -> Result<()> {
        let y = Variable::from(0.5);
        let dy = Variable::from(3.0);
        let f = Softmax::new(1);
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
    fn softmax_normal() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, -2.0, 2.0, 1.0, 0.0, 3.0, 3.0],
            vec![4, 2],
        )?.into());
        let y = softmax(&x, 1)?;
        assert_eq!(*y.shape(), vec![4, 2]);
        Ok(())
    }
}
