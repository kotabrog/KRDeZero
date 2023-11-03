use std::cell::{Ref, RefMut};
use super::{Variable, VariableData};

impl Variable {
    pub fn data(&self) -> Ref<VariableData> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| &inner.data)
    }

    pub fn data_mut(&mut self) -> RefMut<VariableData> {
        let inner = self.inner.borrow_mut();
        RefMut::map(inner, |inner| &mut inner.data)
    }

    pub fn set_data(&mut self, data: VariableData) {
        let mut inner = self.inner.borrow_mut();
        inner.data = data;
    }
}
