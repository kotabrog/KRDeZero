use std::collections::HashMap;
use anyhow::Result;
use crate::{Variable, VariableType, Layer, LayerContent};
use crate::layer::Linear;

pub struct MLP {
    pub layer: Vec<Layer>,
    pub activation: Box<dyn Fn(&Variable) -> Result<Variable>>,
}

impl MLP {
    pub fn new<F>(sizes: &[usize], activation: F) -> Result<Self> 
    where
        F: Fn(&Variable) -> Result<Variable> + 'static
    {
        let mut layer = Vec::new();
        for i in 0..sizes.len() - 1 {
            let l = Layer::new(
                Linear::new(
                    sizes[i],
                    sizes[i + 1],
                    true,
                    VariableType::F64
            )?);
            layer.push(l);
        }
        Ok(Self {
            layer,
            activation: Box::new(activation),
        })
    }
}

impl LayerContent for MLP {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        let x_ref = xs
            .iter()
            .map(|&x| x.clone()).collect::<Vec<_>>();
        let len = self.layer.len();
        let mut y = None;
        for i in 0..len - 1 {
            if let Some(y_content) = y {
                let mut ys = self.layer[i].forward(&[y_content])?;
                y = Some(ys.remove(0));
            } else {
                let mut ys = self.layer[i].forward(&x_ref)?;
                y = Some(ys.remove(0));
            }
            y = Some((self.activation)(&y.unwrap())?);
        }
        let ys = self.layer[len - 1].forward(&[y.unwrap()])?;
        Ok(ys)
    }

    fn get_layers(&self) -> HashMap<String, Layer> {
        let mut layers = HashMap::new();
        for (i, l) in self.layer.iter().enumerate() {
            layers.insert(format!("l{}", i + 1), l.clone());
        }
        layers
    }
}
