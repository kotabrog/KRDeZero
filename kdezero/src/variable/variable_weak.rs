use std::rc::{Rc, Weak};
use std::cell::RefCell;
use super::{Variable, VariableInner};

#[derive(Debug, Clone)]
pub struct VariableWeak {
    inner: Weak<RefCell<VariableInner>>,
}

impl VariableWeak {
    pub fn new(variable: Variable) -> Self {
        Self {
            inner: Rc::downgrade(&variable.inner),
        }
    }

    pub fn upgrade(&self) -> Option<Variable> {
        self.inner.upgrade().map(|inner| Variable { inner })
    }
}
