use anyhow::Result;
use crate::error::KDeZeroError;
use super::Variable;

impl Variable {
    pub fn is_grad_none(&self) -> bool {
        let inner = self.inner.borrow();
        inner.grad.is_none()
    }

    pub fn grad_clone(&self) -> Option<Variable> {
        let inner = self.inner.borrow();
        inner.grad.clone()
    }

    pub fn grad_result(&self) -> Result<Variable> {
        let inner = self.inner.borrow();
        inner.grad.clone().ok_or_else(|| {
            KDeZeroError::Error(inner.name.clone()).into()
        })
    }

    pub fn set_grad(&mut self, grad: Variable) {
        let mut inner = self.inner.borrow_mut();
        inner.grad = Some(grad);
    }

    pub(super) fn set_default_grad_if_none(&mut self) -> Result<()> {
        if self.is_grad_none() {
            let ones = self.data().ones_like()?;
            self.set_grad(ones.into());
        }
        Ok(())
    }

    pub fn clear_grad(&mut self) {
        let mut inner = self.inner.borrow_mut();
        inner.grad = None;
    }
}
