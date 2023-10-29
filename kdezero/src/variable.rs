mod variable_data;

use ktensor::Tensor;

pub use variable_data::VariableData;

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    data: VariableData,
    grad: Option<VariableData>,
}

impl Variable {
    pub fn new(data: VariableData) -> Self {
        Self {
            data,
            grad: None,
        }
    }

    pub fn data(&self) -> &VariableData {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut VariableData {
        &mut self.data
    }

    pub fn set_data(&mut self, data: VariableData) {
        self.data = data;
    }
}

impl From<f32> for Variable {
    fn from(data: f32) -> Self {
        Self::new(data.into())
    }
}

impl From<f64> for Variable {
    fn from(data: f64) -> Self {
        Self::new(data.into())
    }
}

impl From<i32> for Variable {
    fn from(data: i32) -> Self {
        Self::new(data.into())
    }
}

impl From<i64> for Variable {
    fn from(data: i64) -> Self {
        Self::new(data.into())
    }
}

impl From<usize> for Variable {
    fn from(data: usize) -> Self {
        Self::new(data.into())
    }
}

impl From<bool> for Variable {
    fn from(data: bool) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<f32>> for Variable {
    fn from(data: Tensor<f32>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<f64>> for Variable {
    fn from(data: Tensor<f64>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<i32>> for Variable {
    fn from(data: Tensor<i32>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<i64>> for Variable {
    fn from(data: Tensor<i64>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<usize>> for Variable {
    fn from(data: Tensor<usize>) -> Self {
        Self::new(data.into())
    }
}

impl From<Tensor<bool>> for Variable {
    fn from(data: Tensor<bool>) -> Self {
        Self::new(data.into())
    }
}

impl From<VariableData> for Variable {
    fn from(data: VariableData) -> Self {
        Self::new(data)
    }
}
