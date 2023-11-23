use std::collections::HashMap;
use anyhow::Result;
use crate::function::sigmoid;
use crate::{Variable, VariableType, Layer, LayerContent};
use crate::layer::Linear;

pub struct TwoLayerNet {
    pub l1: Layer,
    pub l2: Layer,
}

impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self> {
        let l1 = Layer::new(
            Linear::new(
                input_size,
                hidden_size,
                true,
                VariableType::F64
        )?);
        let l2 = Layer::new(
            Linear::new(
                hidden_size,
                output_size,
                true,
                VariableType::F64
        )?);
        Ok(Self { l1, l2 })
    }
}

impl LayerContent for TwoLayerNet {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        let x_ref = xs
            .iter()
            .map(|&x| x.clone()).collect::<Vec<_>>();
        let mut ys = self.l1.forward(&x_ref)?;
        let y = ys.remove(0);
        let y = sigmoid(&y)?;
        let ys = self.l2.forward(&[y])?;
        Ok(ys)
    }

    fn get_layers(&self) -> HashMap<String, Layer> {
        let mut layers = HashMap::new();
        layers.insert("l1".to_string(), self.l1.clone());
        layers.insert("l2".to_string(), self.l2.clone());
        layers
    }
}
