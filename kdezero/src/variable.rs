mod variable_data;

use std::rc::{Rc, Weak};
use std::cell::{RefCell, Ref, RefMut};
use std::collections::VecDeque;
use anyhow::Result;
use ktensor::Tensor;
use crate::Function;
use crate::error::KDeZeroError;

pub use variable_data::VariableData;

#[derive(Debug, Clone)]
pub struct VariableInner {
    pub data: VariableData,
    pub grad: Option<Variable>,
    pub creator: Option<Function>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct VariableWeak {
    inner: Weak<RefCell<VariableInner>>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    inner: Rc<RefCell<VariableInner>>,
}

impl VariableInner {
    pub fn new(data: VariableData) -> Self {
        Self {
            data,
            grad: None,
            creator: None,
            name: "".to_string(),
        }
    }
}

impl VariableWeak {
    pub fn new(variable: Variable) -> Self {
        Self {
            inner: Rc::downgrade(&variable.inner),
        }
    }

    pub fn upgrade(&self) -> Option<Variable> {
        self.inner.upgrade().map(|inner| Variable { inner })
    }
}

impl Variable {
    pub fn new(data: VariableData) -> Self {
        Self {
            inner: Rc::new(RefCell::new(VariableInner::new(data))),
        }
    }

    pub fn name(&self) -> Ref<String> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| &inner.name)
    }

    pub fn data(&self) -> Ref<VariableData> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| &inner.data)
    }

    pub fn data_mut(&mut self) -> RefMut<VariableData> {
        let inner = self.inner.borrow_mut();
        RefMut::map(inner, |inner| &mut inner.data)
    }

    pub fn set_data(&mut self, data: VariableData) {
        let mut inner = self.inner.borrow_mut();
        inner.data = data;
    }

    pub fn is_grad_none(&self) -> bool {
        let inner = self.inner.borrow();
        inner.grad.is_none()
    }

    pub fn grad_result(&self) -> Result<Variable> {
        let inner = self.inner.borrow();
        inner.grad.clone().ok_or_else(|| {
            KDeZeroError::Error(inner.name.clone()).into()
        })
    }

    pub fn set_grad(&mut self, grad: Variable) {
        let mut inner = self.inner.borrow_mut();
        inner.grad = Some(grad);
    }

    pub fn get_creator_clone(&self) -> Option<Function> {
        let inner = self.inner.borrow();
        inner.creator.clone()
    }

    pub fn get_creator_clone_result(&self) -> Result<Function> {
        let inner = self.inner.borrow();
        inner.creator.clone().ok_or_else(|| {
            KDeZeroError::NoCreator(inner.name.clone()).into()
        })
    }

    pub fn set_creator(&mut self, creator: Function) {
        let mut inner = self.inner.borrow_mut();
        inner.creator = Some(creator);
    }

    pub fn backward(&mut self) -> Result<()> {
        let mut funcs = VecDeque::new();
        funcs.push_back(self.get_creator_clone_result()?);
        while !funcs.is_empty() {
            let f = funcs.pop_front().unwrap();
            let xs = f.inputs_clone_result()?;
            let ys = f.outputs_clone_result()?;
            let grad = ys
                .iter()
                .map(|y| y.grad_result())
                .collect::<Result<Vec<Variable>>>()?;
            let xgs = f.backward(&grad)?;
            for (mut x, xg) in xs.into_iter().zip(xgs) {
                if x.is_grad_none() {
                    x.set_grad(xg);
                } else {
                    // let gx = x.grad_result()?;
                    // x.set_grad(gx.add(&xg)?);
                    todo!()
                }
                if let Some(c) = x.get_creator_clone() {
                    funcs.push_back(c);
                }
            }
        }
        Ok(())
    }
}

impl From<f32> for Variable {
    fn from(data: f32) -> Self {
        Self::new(data.into())
    }
}

impl From<f64> for Variable {
    fn from(data: f64) -> Self {
        Self::new(data.into())
    }
}

impl From<i32> for Variable {
    fn from(data: i32) -> Self {
        Self::new(data.into())
    }
}

impl From<i64> for Variable {
    fn from(data: i64) -> Self {
        Self::new(data.into())
    }
}

impl From<usize> for Variable {
    fn from(data: usize) -> Self {
        Self::new(data.into())
    }
}

impl From<bool> for Variable {
    fn from(data: bool) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<f32>> for Variable {
    fn from(data: Tensor<f32>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<f64>> for Variable {
    fn from(data: Tensor<f64>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<i32>> for Variable {
    fn from(data: Tensor<i32>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<i64>> for Variable {
    fn from(data: Tensor<i64>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<usize>> for Variable {
    fn from(data: Tensor<usize>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<bool>> for Variable {
    fn from(data: Tensor<bool>) -> Self {
        Self::new(data.into())
    }
}

impl From<VariableData> for Variable {
    fn from(data: VariableData) -> Self {
        Self::new(data)
    }
}
