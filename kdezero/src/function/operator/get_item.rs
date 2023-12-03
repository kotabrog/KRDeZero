use anyhow::Result;
use crate::{Variable, VariableData};
use super::super::{FunctionContent, Function};
use crate::utility::check_variable_count;

#[derive(Debug, Clone)]
pub enum SlicePattern {
    Int(usize),
    Vec(Vec<usize>),
    VecVec(Vec<Vec<usize>>),
}

#[derive(Debug)]
pub struct GetItem {
    pub indexes: SlicePattern,
}

impl GetItem {
    pub fn new(indexes: SlicePattern) -> Self {
        Self { indexes }
    }
}

impl FunctionContent for GetItem {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = match self.indexes {
            SlicePattern::Int(i) =>
                x.slice_with_one_index(i)?,
            SlicePattern::Vec(ref v) =>
                x.slice_with_one_indexes(v)?,
            SlicePattern::VecVec(ref v) =>
                x.slice_with_indexes(v.clone())?,
        };
        Ok(vec![y.into()])
    }

    fn backward(&self, xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        check_variable_count(&gys, 1)?;
        let x = xs[0];
        let gy = gys[0];
        let mut func = Function::new(
            GetItemGrad::new(
                self.indexes.clone(),
                x.shape().to_vec()));
        let gxs = func.forward(&[gy.clone()])?;
        Ok(gxs)
    }

    fn name(&self) -> String {
        "GetItem".to_string()
    }
}

#[derive(Debug)]
pub struct GetItemGrad {
    pub indexes: SlicePattern,
    pub in_shape: Vec<usize>,
}

impl GetItemGrad {
    pub fn new(indexes: SlicePattern, in_shape: Vec<usize>) -> Self {
        Self { indexes, in_shape }
    }
}

impl FunctionContent for GetItemGrad {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0].data();
        let y = VariableData::zeros_type(
            &self.in_shape,
            x.get_variable_type())?;
        let y = match self.indexes {
            SlicePattern::Int(i) =>
                y.add_at_one_index(&x, i)?,
            SlicePattern::Vec(ref v) =>
                y.add_at_one_indexes(&x, &v)?,
            SlicePattern::VecVec(ref v) =>
                y.add_at_with_indexes(&x, v.clone())?,
        };
        Ok(vec![y.into()])
    }

    fn backward(&self, _xs: Vec<&Variable>, _ys: Vec<&Variable>, gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&gys, 1)?;
        let gy = gys[0];
        let gx = get_item(&gy, self.indexes.clone())?;
        Ok(vec![gx])
    }

    fn name(&self) -> String {
        "GetItemGrad".to_string()
    }
}

pub fn get_item(x: &Variable, indexes: SlicePattern) -> Result<Variable> {
    let mut func = Function::new(GetItem::new(indexes));
    let mut ys = func.forward(&[x.clone()])?;
    let y = ys.remove(0);
    Ok(y)
}

pub fn get_item_with_one_index(x: &Variable, index: usize) -> Result<Variable> {
    get_item(x, SlicePattern::Int(index))
}

pub fn get_item_with_one_indexes(x: &Variable, indexes: &[usize]) -> Result<Variable> {
    get_item(x, SlicePattern::Vec(indexes.to_vec()))
}

pub fn get_item_with_indexes(x: &Variable, indexes: &[Vec<usize>]) -> Result<Variable> {
    get_item(x, SlicePattern::VecVec(indexes.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;
    use ktensor::Tensor;

    #[test]
    fn get_item_forward_int() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        let y = GetItem::new(SlicePattern::Int(1)).forward(vec![&x])?;
        assert_eq!(*y[0].data(), Tensor::new(
            vec![2.0, 3.0],
            vec![2],
        )?.into());
        Ok(())
    }

    #[test]
    fn get_item_forward_vec() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        let y = GetItem::new(SlicePattern::Vec(vec![1, 0])).forward(vec![&x])?;
        assert_eq!(*y[0].data(), Tensor::new(
            vec![2.0, 3.0, 0.0, 1.0],
            vec![2, 2],
        )?.into());
        Ok(())
    }

    #[test]
    fn get_item_forward_vecvec() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![2, 2, 2],
        )?.into());
        let y = GetItem::new(SlicePattern::VecVec(vec![vec![1, 0], vec![0, 1]])).forward(vec![&x])?;
        assert_eq!(*y[0].data(), Tensor::new(
            vec![4.0, 5.0, 2.0, 3.0],
            vec![2, 2],
        )?.into());
        Ok(())
    }

    #[test]
    fn error_get_item_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        match GetItem::new(SlicePattern::Int(1)).forward(vec![&x.clone(), &x]) {
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
    fn get_item_backward_int() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![6.0, 7.0],
            vec![2],
        )?.into());
        let f = GetItem::new(SlicePattern::Int(1));
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), Tensor::new(
            vec![0.0, 0.0, 6.0, 7.0, 0.0, 0.0],
            vec![3, 2],
        )?.into());
        Ok(())
    }

    #[test]
    fn get_item_backward_vec() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![6.0, 7.0, 8.0, 9.0],
            vec![2, 2],
        )?.into());
        let f = GetItem::new(SlicePattern::Vec(vec![1, 0]));
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), Tensor::new(
            vec![8.0, 9.0, 6.0, 7.0, 0.0, 0.0],
            vec![3, 2],
        )?.into());
        Ok(())
    }

    #[test]
    fn get_item_backward_vecvec() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![2, 2, 2],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![7.0, 8.0, 9.0, 10.0],
            vec![2, 2],
        )?.into());
        let f = GetItem::new(SlicePattern::VecVec(vec![vec![1, 0], vec![0, 1]]));
        let dx = f.backward(vec![&x], vec![], vec![&dy])?;
        assert_eq!(*dx[0].data(), Tensor::new(
            vec![0.0, 0.0, 9.0, 10.0, 7.0, 8.0, 0.0, 0.0],
            vec![2, 2, 2],
        )?.into());
        Ok(())
    }

    #[test]
    fn error_get_item_backward_invalid_variable_count_dy() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![6.0, 7.0],
            vec![2],
        )?.into());
        let f = GetItem::new(SlicePattern::Int(1));
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
    fn error_get_item_backward_invalid_variable_count_x() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        let dy = Variable::new(Tensor::new(
            vec![6.0, 7.0],
            vec![2],
        )?.into());
        let f = GetItem::new(SlicePattern::Int(1));
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
    fn get_item_normal() -> Result<()> {
        let x = Variable::new(Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![3, 2],
        )?.into());
        let y = get_item(&x, SlicePattern::Int(1))?;
        assert_eq!(*y.data(), Tensor::new(
            vec![2.0, 3.0],
            vec![2],
        )?.into());
        Ok(())
    }
}
