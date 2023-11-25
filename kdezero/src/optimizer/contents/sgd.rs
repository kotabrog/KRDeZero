use anyhow::Result;
use crate::Variable;
use super::super::OptimizerContent;

pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl OptimizerContent for SGD {
    fn update_one(&mut self, param: &mut Variable) -> Result<()> {
        let new_data = param.data()
            .sub(&param.grad_result()?.data().scalar_mul(self.lr)?)?;
        param.set_data(new_data);
        Ok(())
    }
}
