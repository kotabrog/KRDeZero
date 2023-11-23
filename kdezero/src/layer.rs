mod contents;

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::{RefCell, Ref};
use anyhow::Result;
use crate::{Variable, VariableWeak};

pub use contents::Linear;

pub trait LayerContent {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>>;

    fn get_params(&self) -> HashMap<String, Variable> {
        HashMap::new()
    }

    fn get_layers(&self) -> HashMap<String, Layer> {
        HashMap::new()
    }

    fn get_params_recursive(&self) -> HashMap<String, Variable> {
        let mut params = self.get_params();
        let layers = self.get_layers();
        for (layer_name, layer) in layers {
            let mut layer_params = layer.get_params_recursive();
            for (name, param) in layer_params.drain() {
                params.insert(format!("{}.{}", layer_name, name), param);
            }
        }
        params
    }
}

pub struct LayerInner {
    pub layer: Box<dyn LayerContent>,
    pub inputs: Option<Vec<VariableWeak>>,
    pub outputs: Option<Vec<VariableWeak>>,
    pub name: String,
}

#[derive(Clone)]
pub struct Layer {
    inner: Rc<RefCell<LayerInner>>,
}

impl LayerInner {
    pub fn new<T>(layer: T) -> Self
    where
        T: LayerContent + 'static
    {
        Self {
            layer: Box::new(layer),
            inputs: None,
            outputs: None,
            name: "".to_string(),
        }
    }
}

impl Layer {
    pub fn new<T>(layer: T) -> Self
    where
        T: LayerContent + 'static
    {
        Self {
            inner: Rc::new(RefCell::new(LayerInner::new(layer))),
        }
    }

    pub fn name(&self) -> Ref<String> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| &inner.name)
    }

    pub fn forward(&self, xs: &[Variable]) -> Result<Vec<Variable>> {
        let xs = xs
            .iter()
            .map(|x| x.clone())
            .collect::<Vec<_>>();
        let refs = xs
            .iter()
            .map(|x| x)
            .collect::<Vec<_>>();
        let inner = &mut self.inner.borrow_mut();
        let ys = inner.layer.forward(refs)?;
        inner.inputs = Some(xs
            .iter()
            .map(|x| VariableWeak::new(x.clone()))
            .collect::<Vec<_>>());
        inner.outputs = Some(ys
            .iter()
            .map(|y| VariableWeak::new(y.clone()))
            .collect::<Vec<_>>());
        Ok(ys)
    }

    pub fn clear_grads(&mut self) {
        let inner = &mut self.inner.borrow();
        let params = inner.layer.get_params();
        for (_, mut param) in params {
            param.clear_grad();
        }
    }

    pub fn clear_grads_recursive(&mut self) {
        let inner = &mut self.inner.borrow();
        let params = inner.layer.get_params_recursive();
        for (_, mut param) in params {
            param.clear_grad();
        }
    }

    pub fn get_params(&self) -> HashMap<String, Variable> {
        let inner = self.inner.borrow();
        inner.layer.get_params()
    }

    pub fn get_params_recursive(&self) -> HashMap<String, Variable> {
        let inner = self.inner.borrow();
        inner.layer.get_params_recursive()
    }
}
