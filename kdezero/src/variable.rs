mod variable_data;
mod variable_weak;
mod name;
mod data;
mod grad;
mod creator;
mod conversion;
mod info;
mod ops;
mod backward;

use std::rc::Rc;
use std::cell::RefCell;
use crate::Function;

pub use variable_data::VariableData;
pub use variable_weak::VariableWeak;

#[derive(Debug, Clone)]
pub struct VariableInner {
    pub data: VariableData,
    pub grad: Option<Variable>,
    pub creator: Option<Function>,
    pub name: String,
    pub generation: usize,
}

#[derive(Debug, Clone)]
pub struct Variable {
    inner: Rc<RefCell<VariableInner>>,
}

impl VariableInner {
    pub fn new(data: VariableData) -> Self {
        Self {
            data,
            grad: None,
            creator: None,
            name: "".to_string(),
            generation: 0,
        }
    }
}

impl Variable {
    pub fn new(data: VariableData) -> Self {
        Self {
            inner: Rc::new(RefCell::new(VariableInner::new(data))),
        }
    }

    pub fn new_with_name(data: VariableData, name: &str) -> Self {
        let mut variable = Self::new(data);
        variable.set_name(name);
        variable
    }

    pub(crate) fn generation(&self) -> usize {
        let inner = self.inner.borrow();
        inner.generation
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.borrow();
        let s = format!("Variable({})", inner.data)
            .replace("\n", &format!("\n{}", " ".repeat(9)));
        write!(f, "{}", s)
    }
}
