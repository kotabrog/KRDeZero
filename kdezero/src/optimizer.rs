mod contents;

use anyhow::Result;
use crate::{Variable, Model};

pub use contents::{SGD, MomentumSGD};

pub trait OptimizerContent {
    fn update_one(&mut self, _param: &mut Variable) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, model: &mut Model) -> Result<()> {
        for (_, param) in model.get_params().iter_mut() {
            self.update_one(param)?;
        }
        Ok(())
    }
}

pub struct Optimizer {
    optimizer: Box<dyn OptimizerContent>,
    model: Option<Model>,
}

impl Optimizer {
    pub fn new<T>(optimizer: T) -> Self
    where
        T: OptimizerContent + 'static
    {
        Self {
            optimizer: Box::new(optimizer),
            model: None,
        }
    }

    pub fn get_model_mut_result(&mut self) -> Result<&mut Model> {
        if let Some(model) = &mut self.model {
            Ok(model)
        } else {
            Err(anyhow::anyhow!("Model is not set"))
        }
    }

    pub fn set_model(&mut self, model: Model) {
        self.model = Some(model);
    }

    pub fn update(&mut self) -> Result<()> {
        if let Some(model) = &mut self.model {
            self.optimizer.update(model)
        } else {
            Ok(())
        }
    }
}
