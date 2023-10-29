mod operator;
mod function_helper;

use std::rc::Rc;
use anyhow::Result;
use crate::Variable;
use crate::error::KDeZeroError;

pub use operator::{
    Square, Exp
};

pub trait Function {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>>;
    fn backward(&self, _xs: Vec<&Variable>, _gys: Vec<&Variable>) -> Result<Vec<Variable>> {
        unimplemented!("backward is not implemented")
    }
}

pub struct FunctionWrapper {
    func: Box<dyn Function>,
    inputs: Option<Vec<Rc<Variable>>>,
    name: String,
}

impl FunctionWrapper {
    pub fn new<T>(func: T) -> Self
    where
        T: Function + 'static
    {
        Self {
            func: Box::new(func),
            inputs: None,
            name: "".to_string(),
        }
    }

    pub fn forward(&mut self, xs: &[Rc<Variable>]) -> Result<Vec<Variable>> {
        let xs = xs
            .iter()
            .map(|x| x.clone())
            .collect::<Vec<_>>();
        let refs = xs
            .iter()
            .map(|x| x.as_ref())
            .collect::<Vec<_>>();
        let ys = self.func.forward(refs)?;
        self.inputs = Some(xs);
        Ok(ys)
    }

    pub fn forward_into(&mut self, xs: Vec<Variable>) -> Result<Vec<Variable>> {
        self.forward(
            &xs
                .iter()
                .map(|x| Rc::new(x.clone()))
                .collect::<Vec<_>>()
        )
    }

    pub fn backward(&self, gys: &[Rc<Variable>]) -> Result<Vec<Variable>> {
        let xs = self.inputs.as_ref()
            .ok_or(KDeZeroError::NoInputVariable(self.name.clone()))?;
        let xs = xs
            .iter()
            .map(|x| x.as_ref())
            .collect::<Vec<_>>();
        let gys = gys
            .iter()
            .map(|x| x.as_ref())
            .collect::<Vec<_>>();
        let gxs = self.func.backward(xs, gys)?;
        Ok(gxs)
    }

    pub fn backward_into(&self, gys: Vec<Variable>) -> Result<Vec<Variable>> {
        self.backward(
            &gys
                .iter()
                .map(|x| Rc::new(x.clone()))
                .collect::<Vec<_>>()
        )
    }
}
