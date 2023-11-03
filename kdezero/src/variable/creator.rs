use anyhow::Result;
use crate::error::KDeZeroError;
use crate::Function;
use super::Variable;

impl Variable {
    pub fn get_creator_clone(&self) -> Option<Function> {
        let inner = self.inner.borrow();
        inner.creator.clone()
    }

    pub fn get_creator_clone_result(&self) -> Result<Function> {
        let inner = self.inner.borrow();
        inner.creator.clone().ok_or_else(|| {
            KDeZeroError::NoCreator(inner.name.clone()).into()
        })
    }

    pub(crate) fn set_creator(&mut self, creator: Function) {
        let mut inner = self.inner.borrow_mut();
        inner.generation = creator.generation() + 1;
        inner.creator = Some(creator);
    }
}
