mod contents;

use std::collections::HashMap;
use std::cell::Ref;
use anyhow::Result;
use crate::{Variable, Layer, LayerContent, plot_dot_graph};

pub use contents::{TwoLayerNet, MLP};

pub struct Model {
    layer: Layer,
}

impl Model {
    pub fn new<T>(layer: T) -> Self
    where
        T: LayerContent + 'static
    {
        Self {
            layer: Layer::new(layer),
        }
    }

    pub fn name(&self) -> Ref<String> {
        self.layer.name()
    }

    pub fn forward(&self, xs: &[Variable]) -> Result<Vec<Variable>> {
        self.layer.forward(xs)
    }

    pub fn clear_grads(&mut self) {
        self.layer.clear_grads_recursive();
    }

    pub fn get_params(&self) -> HashMap<String, Variable> {
        self.layer.get_params_recursive()
    }

    pub fn plot(&self, inputs: &[Variable], out_path_without_extension: &str) -> Result<()> {
        let ys = self.forward(inputs)?;
        plot_dot_graph(&ys[0], out_path_without_extension, true, true)
    }
}
