pub mod operator;
mod function_helper;

use anyhow::Result;
use crate::Variable;

pub trait Function {
    fn forward(&self, xs: Vec<Variable>) -> Result<Vec<Variable>>;
}
