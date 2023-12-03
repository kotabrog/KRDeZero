use anyhow::Result;
use crate::Variable;
use super::{sub, mul, broadcast_to, softmax};
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug)]
pub struct SoftmaxCrossEntropy {}

impl SoftmaxCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl FunctionContent for SoftmaxCrossEntropy {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        let x = xs[0].data();
        let t = xs[1].data()
            .to_usize_tensor()?
            .to_vector()?;
        let n = x.shape()[0];
        let log_z = x.log_sum_exp(1)?
            .broadcast_to(&x.shape())?;
        let log_p = x.sub(&log_z)?;
        let log_p = log_p.slice_with_indexes(vec![
            (0..n).collect(),
            t,
        ])?;
        let y = log_p.sum(None, false)?
            .neg()?
            .scalar_mul(1. / (n as f64))?;
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 2)?;
        check_variable_count(&gys, 1)?;
        let gy = gys[0];
        let x = xs[0];
        let t = xs[1];
        let n = x.shape()[0];
        let class_num = x.shape()[1];
        let scalar = gy.data().full_like(1. / (n as f64))?;
        let gy = mul(&gy, &scalar.into())?;
        let y = softmax(x, 1)?;
        let t_onehot = y.data().eye_like_type(class_num)?
            .slice_with_one_indexes(&t.data()
                .to_usize_tensor()?.to_vector()?)?;
        let y = sub(&y, &t_onehot.into())?;
        let gx = mul(
            &y,
            &broadcast_to(&gy, &y.shape())?)?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "SoftmaxCrossEntropy".to_string()
    }
}

pub fn softmax_cross_entropy(x: &Variable, t: &Variable) -> Result<Variable> {
    let mut func = Function::new(SoftmaxCrossEntropy::new());
    let mut ys = func.forward(&[x.clone(), t.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;
    use ktensor::Tensor;

    #[test]
    fn softmax_cross_entropy_forward() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.1, 0.9, 0.5, 0.5, 0.9, 0.1],
            vec![3, 2],
        )?.into());
        let t = Variable::new(Tensor::<usize>::new(
            vec![0, 1, 0],
            vec![3],
        )?.into());
        let y = SoftmaxCrossEntropy::new().forward(vec![&x, &t])?;
        let y_data = y[0].data().clone();
        assert_eq!(y_data.shape(), &vec![]);
        Ok(())
    }

    #[test]
    fn softmax_cross_entropy_forward_value_check() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.1, 0.9],
            vec![1, 2],
        )?.into());
        let t = Variable::new(Tensor::<usize>::new(
            vec![0],
            vec![1],
        )?.into());
        let y = SoftmaxCrossEntropy::new().forward(vec![&x, &t])?;
        let y_data = y[0].data().clone();
        assert_eq!(y_data.shape(), &vec![]);
        assert!(y_data.to_f64_tensor()?.get_data()[0] > 1.);
        Ok(())
    }

    #[test]
    fn error_softmax_cross_entropy_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.1, 0.9],
            vec![1, 2],
        )?.into());
        let t = Variable::new(Tensor::<usize>::new(
            vec![0],
            vec![1],
        )?.into());
        match SoftmaxCrossEntropy::new().forward(vec![&x.clone(), &t, &x]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    2,
                    3,
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn softmax_cross_entropy_backward() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.1, 0.9, 0.5, 0.5, 0.9, 0.1],
            vec![3, 2],
        )?.into());
        let t = Variable::new(Tensor::<usize>::new(
            vec![0, 1, 0],
            vec![3],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![1.5],
            vec![],
        )?.into());
        let f = SoftmaxCrossEntropy::new();
        let dx = f.backward(vec![&x, &t], vec![], vec![&dy])?;
        assert_eq!(*dx[0].shape(), vec![3, 2]);
        Ok(())
    }

    #[test]
    fn error_softmax_cross_entropy_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.1, 0.9, 0.5, 0.5, 0.9, 0.1],
            vec![3, 2],
        )?.into());
        let t = Variable::new(Tensor::<usize>::new(
            vec![0, 1, 0],
            vec![3],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![1.5],
            vec![],
        )?.into());
        let f = SoftmaxCrossEntropy::new();
        match f.backward(vec![&x, &t], vec![], vec![&dy, &dy]) {
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
    fn error_softmax_cross_entropy_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.1, 0.9, 0.5, 0.5, 0.9, 0.1],
            vec![3, 2],
        )?.into());
        let t = Variable::new(Tensor::<usize>::new(
            vec![0, 1, 0],
            vec![3],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![1.5],
            vec![],
        )?.into());
        let f = SoftmaxCrossEntropy::new();
        match f.backward(vec![&x, &t, &x], vec![], vec![&dy]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    2,
                    3,
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn softmax_cross_entropy_normal() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, -2.0, 2.0, 1.0, 0.0, 3.0, 3.0],
            vec![4, 2],
        )?.into());
        let t = Variable::new(Tensor::<usize>::new(
            vec![0, 1, 0, 1],
            vec![4],
        )?.into());
        let y = softmax_cross_entropy(&x, &t)?;
        assert_eq!(*y.shape(), vec![]);
        Ok(())
    }
}
