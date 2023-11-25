use std::collections::HashMap;
use anyhow::Result;
use crate::Variable;
use super::super::OptimizerContent;

pub struct MomentumSGD {
    pub lr: f64,
    pub momentum: f64,
    pub params: HashMap<usize, Variable>,
}

impl MomentumSGD {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            params: HashMap::new(),
        }
    }
}

impl OptimizerContent for MomentumSGD {
    fn update_one(&mut self, param: &mut Variable) -> Result<()> {
        let id = param.id();
        if !self.params.contains_key(&id) {
            self.params.insert(
                id,
                param.data().zeros_like()?.into()
            );
        }
        let v = self.params.get_mut(&id).unwrap();
        let new_v = v.data()
            .scalar_mul(self.momentum)?
            .sub(&param.grad_result()?.data().scalar_mul(self.lr)?)?;
        v.set_data(new_v);
        let new_data = param.data().add(&v.data())?;
        param.set_data(new_data);
        Ok(())
    }
}
