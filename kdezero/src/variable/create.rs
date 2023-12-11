use super::{Variable, VariableData};

impl Variable {
    pub fn none() -> Self {
        Self::new(VariableData::none())
    }
}
