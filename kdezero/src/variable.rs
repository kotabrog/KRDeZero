mod variable_data;

use std::rc::{Rc, Weak};
use std::cell::{RefCell, Ref, RefMut};
use std::collections::{BinaryHeap, HashSet};
use anyhow::Result;
use ktensor::Tensor;
use crate::Function;
use crate::error::KDeZeroError;
use crate::function::add;

pub use variable_data::VariableData;

#[derive(Debug, Clone)]
pub struct VariableInner {
    pub data: VariableData,
    pub grad: Option<Variable>,
    pub creator: Option<Function>,
    pub name: String,
    pub generation: usize,
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
            generation: 0,
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

    pub fn new_with_name(data: VariableData, name: &str) -> Self {
        let mut variable = Self::new(data);
        variable.set_name(name);
        variable
    }

    pub fn name(&self) -> Ref<String> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| &inner.name)
    }

    pub fn set_name(&mut self, name: &str) {
        let mut inner = self.inner.borrow_mut();
        inner.name = name.to_string();
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

    pub fn grad_clone(&self) -> Option<Variable> {
        let inner = self.inner.borrow();
        inner.grad.clone()
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

    pub fn clear_grad(&mut self) {
        let mut inner = self.inner.borrow_mut();
        inner.grad = None;
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
        inner.generation = creator.generation() + 1;
        inner.creator = Some(creator);
    }

    pub fn generation(&self) -> usize {
        let inner = self.inner.borrow();
        inner.generation
    }

    pub fn shape(&self) -> Ref<[usize]> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| inner.data.shape())
    }

    pub fn ndim(&self) -> usize {
        let inner = self.inner.borrow();
        inner.data.ndim()
    }

    pub fn size(&self) -> usize {
        let inner = self.inner.borrow();
        inner.data.size()
    }

    pub fn data_type(&self) -> Ref<str> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| inner.data.data_type())
    }

    fn backward_inner(&mut self, retain_grad: bool) -> Result<()> {
        if self.is_grad_none() {
            let ones = self.data().ones_like()?;
            self.set_grad(ones.into());
        }

        struct OrdFunction {
            pub function: Function,
            pub generation: usize,
        }

        impl PartialEq for OrdFunction {
            fn eq(&self, other: &Self) -> bool {
                self.generation == other.generation
            }
        }

        impl Eq for OrdFunction {}

        impl PartialOrd for OrdFunction {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.generation.partial_cmp(&other.generation)
            }
        }

        impl Ord for OrdFunction {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.generation.cmp(&other.generation)
            }
        }

        let mut funcs = BinaryHeap::new();
        let mut seen_set = HashSet::new();

        fn add_func(func: &Function, funcs: &mut BinaryHeap<OrdFunction>, seen_set: &mut HashSet<Function>) {
            if seen_set.contains(func) {
                return;
            }
            seen_set.insert(func.clone());
            let generation = func.generation();
            funcs.push(OrdFunction { function: func.clone(), generation });
        }

        add_func(&self.get_creator_clone_result()?, &mut funcs, &mut seen_set);

        while !funcs.is_empty() {
            let f = funcs.pop().unwrap()
                .function;
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
                    x.set_grad(add(&x.grad_result()?, &xg)?);
                }
                if let Some(c) = x.get_creator_clone() {
                    add_func(&c, &mut funcs, &mut seen_set);
                }
            }
            if !retain_grad {
                for mut y in f.outputs_clone_result()? {
                    y.clear_grad();
                }
            }
        }
        Ok(())
    }

    pub fn backward(&mut self) -> Result<()> {
        self.backward_inner(false)
    }

    pub fn backward_retain_grad(&mut self) -> Result<()> {
        self.backward_inner(true)
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

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.borrow();
        let s = format!("Variable({})", inner.data)
            .replace("\n", &format!("\n{}", " ".repeat(9)));
        write!(f, "{}", s)
    }
}
