mod operator;
mod function_helper;

use std::rc::Rc;
use std::cell::{RefCell, Ref};
use anyhow::Result;
use crate::{Variable, VariableWeak};
use crate::error::KDeZeroError;

pub use operator::{
    Square, Exp
};

pub trait FunctionContent: std::fmt::Debug {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>>;
    fn backward(&self, _xs: Vec<&Variable>, _gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        unimplemented!("backward is not implemented")
    }
}

#[derive(Debug)]
pub struct FunctionInner {
    pub func: Box<dyn FunctionContent>,
    pub inputs: Option<Vec<Variable>>,
    pub outputs: Option<Vec<VariableWeak>>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Function {
    inner: Rc<RefCell<FunctionInner>>,
}

impl FunctionInner {
    pub fn new<T>(func: T) -> Self
    where
        T: FunctionContent + 'static
    {
        Self {
            func: Box::new(func),
            inputs: None,
            outputs: None,
            name: "".to_string(),
        }
    }
}

impl Function {
    pub fn new<T>(func: T) -> Self
    where
        T: FunctionContent + 'static
    {
        Self {
            inner: Rc::new(RefCell::new(FunctionInner::new(func))),
        }
    }

    pub fn name(&self) -> Ref<String> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| &inner.name)
        
    }

    pub fn inputs_clone_result(&self) -> Result<Vec<Variable>> {
        let inner = &self.inner.borrow();
        let inputs = inner.inputs.as_ref()
            .ok_or(KDeZeroError::NoInputVariable(inner.name.clone()))?;
        let inputs = inputs
            .iter()
            .map(|x| x.clone())
            .collect::<Vec<_>>();
        Ok(inputs)
    }

    pub fn outputs_clone_result(&self) -> Result<Vec<Variable>> {
        let inner = &self.inner.borrow();
        let outputs = inner.outputs.as_ref()
            .ok_or(KDeZeroError::NoOutputVariable(inner.name.clone()))?;
        let outputs = outputs
            .iter()
            .map(|x| x.upgrade())
            .collect::<Option<Vec<_>>>()
            .ok_or(KDeZeroError::NoOutputVariable(inner.name.clone()))?;
        Ok(outputs)
    }

    pub fn forward(&mut self, xs: &[Variable]) -> Result<Vec<Variable>> {
        let inner = &mut self.inner.borrow_mut();
        let xs = xs
            .iter()
            .map(|x| x.clone())
            .collect::<Vec<_>>();
        let refs = xs
            .iter()
            .map(|x| x)
            .collect::<Vec<_>>();
        let mut ys = inner.func.forward(refs)?;
        for y in &mut ys {
            y.set_creator(self.clone());
        }
        inner.outputs = Some(
            ys
                .iter()
                .map(|y| VariableWeak::new(y.clone()))
                .collect::<Vec<_>>()
        );
        inner.inputs = Some(xs);
        Ok(ys)
    }

    pub fn backward(&self, gys: &[Variable]) -> Result<Vec<Variable>> {
        let inner = &mut self.inner.borrow_mut();
        let xs = inner.inputs.as_ref()
            .ok_or(KDeZeroError::NoInputVariable(inner.name.clone()))?;
        let xs = xs
            .iter()
            .map(|x| x)
            .collect::<Vec<_>>();
        let gys = gys
            .iter()
            .map(|x| x)
            .collect::<Vec<_>>();
        let gxs = inner.func.backward(xs, gys)?;
        Ok(gxs)
    }
}
