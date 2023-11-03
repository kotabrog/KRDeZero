use std::cell::Ref;
use super::Variable;

impl Variable {
    pub fn name(&self) -> Ref<String> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| &inner.name)
    }

    pub fn set_name(&mut self, name: &str) {
        let mut inner = self.inner.borrow_mut();
        inner.name = name.to_string();
    }
}
