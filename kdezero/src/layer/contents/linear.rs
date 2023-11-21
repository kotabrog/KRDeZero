use std::collections::HashMap;
use anyhow::Result;
use crate::{Variable, VariableType, VariableData};
use super::super::LayerContent;
use crate::utility::{check_variable_count, check_dimensions};
use crate::function::{self, broadcast_to};
use crate::error::KDeZeroError;

#[derive(Debug)]
pub struct Linear {
    pub weight: Variable,
    pub bias: Option<Variable>,
    pub in_size: usize,
    pub out_size: usize,
    pub variable_type: VariableType,
}

impl Linear {
    pub fn new(in_size: usize, out_size: usize, bias: bool, variable_type: VariableType) -> Result<Self> {
        match variable_type {
            VariableType::F32 | VariableType::F64 => (),
            _ => return Err(KDeZeroError::NotImplementedType(
                variable_type.to_string(),
                "Linear".to_string(),
            ).into()),
        }
        let weight = VariableData::random_normal(&[in_size, out_size], variable_type)?.into();
        let bias = if bias {
            Some(VariableData::zeros_type(&[out_size], variable_type)?.into())
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_size,
            out_size,
            variable_type,
        })
    }
}

impl LayerContent for Linear {
    fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
        check_variable_count(&xs, 1)?;
        let x = xs[0];
        check_dimensions(x, 2)?;
        let w = &self.weight;
        let b_shape = [x.shape()[0], self.out_size];
        let y = function::linear(x, w, self.bias
            .as_ref()
            .map(|b| broadcast_to(b, &b_shape))
            .transpose()?
            .as_ref()
        )?;
        Ok(vec![y])
    }

    fn get_params(&self) -> HashMap<String, Variable> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(bias) = &self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }
}
