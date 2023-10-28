mod operator;
mod function_helper;

use anyhow::Result;
use crate::Variable;

pub use operator::{
    Square, Exp
};

pub trait Function {
    fn forward(&self, xs: &[Variable]) -> Result<Vec<Variable>>;
}
