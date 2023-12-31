use anyhow::Result;
use num_traits::{One, Zero};
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: PartialOrd + Clone,
{
    /// Get the maximum value of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<T>` - The maximum value
    /// 
    /// # Note
    /// 
    /// If the tensor is empty, `EmptyTensorError` is returned
    pub fn max(&self) -> Result<T> {
        if self.data.len() == 0 {
            Err(TensorError::EmptyTensorError().into())
        } else {
            let mut max = self.data[0].clone();
            for i in 1..self.data.len() {
                if max < self.data[i] {
                    max = self.data[i].clone();
                }
            }
            Ok(max)
        }
    }

    /// Get the minimum value of the tensor with the axis
    /// 
    /// # Arguments
    /// 
    /// * `axis` - The axis
    /// * `keepdims` - Whether to keep the dimension
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - The minimum value
    /// 
    /// # Note
    /// 
    /// If the tensor is empty, `EmptyTensorError` is returned
    /// If axis is out of range, `DimensionLargerError` is returned
    pub fn max_with_axis(&self, axis: usize, keepdims: bool) -> Result<Self> {
        if self.data.len() == 0 {
            return Err(TensorError::EmptyTensorError().into())
        }
        if axis >= self.ndim() {
            return Err(TensorError::DimensionLargerError(axis, self.ndim() - 1).into())
        }
        let mut shape = self.shape.clone();
        if keepdims {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        let size = self.shape[axis];
        let data_size = self.size() / size;
        let before_axis_size: usize = self.shape[..axis].iter().product();
        let after_axis_size = data_size / before_axis_size;
        let mut new_data = Vec::with_capacity(data_size);
        let mut after = 0;
        let mut before = 0;
        for _ in 0..data_size {
            let index = before * after_axis_size * size + after;
            let mut max = self.data[index].clone();
            for j in 1..size {
                let value = &self.data[index + j * after_axis_size];
                if max < *value {
                    max = value.clone();
                }
            }
            new_data.push(max);
            after += 1;
            if after == after_axis_size {
                after = 0;
                before += 1;
            }
        }
        Tensor::new(new_data, shape)
    }

    /// Get the maximum value index of the tensor with the axis
    /// 
    /// # Arguments
    /// 
    /// * `axis` - The axis
    /// * `keepdims` - Whether to keep the dimension
    /// 
    /// # Returns
    /// 
    /// * `Result<Tensor<usize>>` - The maximum value index
    /// 
    /// # Note
    /// 
    /// If there are multiple maximum values, the first index is returned
    /// If the tensor is empty, `EmptyTensorError` is returned
    /// If axis is out of range, `DimensionLargerError` is returned
    pub fn argmax_with_axis(&self, axis: usize, keepdims: bool) -> Result<Tensor<usize>> {
        if self.data.len() == 0 {
            return Err(TensorError::EmptyTensorError().into())
        }
        if axis >= self.ndim() {
            return Err(TensorError::DimensionLargerError(axis, self.ndim() - 1).into())
        }
        let mut shape = self.shape.clone();
        if keepdims {
            shape[axis] = 1;
        } else {
            shape.remove(axis);
        }
        let size = self.shape[axis];
        let data_size = self.size() / size;
        let before_axis_size: usize = self.shape[..axis].iter().product();
        let after_axis_size = data_size / before_axis_size;
        let mut new_data = Vec::with_capacity(data_size);
        let mut after = 0;
        let mut before = 0;
        for _ in 0..data_size {
            let index = before * after_axis_size * size + after;
            let mut max = self.data[index].clone();
            let mut max_index = 0;
            for j in 1..size {
                let value = &self.data[index + j * after_axis_size];
                if max < *value {
                    max = value.clone();
                    max_index = j;
                }
            }
            new_data.push(max_index);
            after += 1;
            if after == after_axis_size {
                after = 0;
                before += 1;
            }
        }
        Tensor::new(new_data, shape)
    }

    /// Get the maximum value of two tensors
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - The maximum value
    /// 
    /// # Note
    /// 
    /// If the shape of the two tensors is different, `ShapeMismatchError` is returned
    pub fn maximum(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatchError(self.shape.clone(), other.shape.clone()).into())
        }
        let data = self.data.iter()
            .zip(other.data.iter())
            .map(|(x, y)| {
                if x > y {
                    x.clone()
                } else {
                    y.clone()
                }
            }).collect::<Vec<_>>();
        Ok(Tensor::new(data, self.shape.clone())?)
    }

    /// Get the minimum value of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Result<T>` - The minimum value
    /// 
    /// # Note
    /// 
    /// If the tensor is empty, `EmptyTensorError` is returned
    pub fn min(&self) -> Result<T> {
        if self.data.len() == 0 {
            Err(TensorError::EmptyTensorError().into())
        } else {
            let mut min = self.data[0].clone();
            for i in 1..self.data.len() {
                if min > self.data[i] {
                    min = self.data[i].clone();
                }
            }
            Ok(min)
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + One + Zero
{
    /// Create a binary mask from the condition
    /// 
    /// # Arguments
    /// 
    /// * `condition` - The condition
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - The binary mask
    pub fn create_binyary_mask_from_condition(&self, condition: fn (&T) -> bool) -> Result<Self> {
        let data = self.data.iter()
            .map(|x| {
                if condition(x) {
                    T::one()
                } else {
                    T::zero()
                }
            }).collect::<Vec<_>>();
        Ok(Tensor::new(data, self.shape.clone())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert_eq!(x.max().unwrap(), 2.0);
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1]).unwrap();
        assert_eq!(x.max().unwrap(), 2.0);
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1, 1]).unwrap();
        assert_eq!(x.max().unwrap(), 2.0);
    }

    #[test]
    fn max_empty() {
        let x = Tensor::<f32>::new([], [0]).unwrap();
        match x.max() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::EmptyTensorError());
            }
        }
    }

    #[test]
    fn max_with_axis_normal() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [2, 3, 2]).unwrap();
        let y = x.max_with_axis(0, false).unwrap();
        assert_eq!(y.get_data(), &vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        assert_eq!(y.get_shape(), &vec![3, 2]);
        let y = x.max_with_axis(0, true).unwrap();
        assert_eq!(y.get_data(), &vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        assert_eq!(y.get_shape(), &vec![1, 3, 2]);
        let y = x.max_with_axis(1, false).unwrap();
        assert_eq!(y.get_data(), &vec![4.0, 5.0, 10.0, 11.0]);
        assert_eq!(y.get_shape(), &vec![2, 2]);
        let y = x.max_with_axis(1, true).unwrap();
        assert_eq!(y.get_data(), &vec![4.0, 5.0, 10.0, 11.0]);
        assert_eq!(y.get_shape(), &vec![2, 1, 2]);
        let y = x.max_with_axis(2, false).unwrap();
        assert_eq!(y.get_data(), &vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
        assert_eq!(y.get_shape(), &vec![2, 3]);
        let y = x.max_with_axis(2, true).unwrap();
        assert_eq!(y.get_data(), &vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
        assert_eq!(y.get_shape(), &vec![2, 3, 1]);
    }

    #[test]
    fn max_with_axis_error_empty() {
        let x = Tensor::<f32>::new([], [0]).unwrap();
        match x.max_with_axis(0, false) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::EmptyTensorError());
            }
        }
    }

    #[test]
    fn max_with_axis_error_axis() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        match x.max_with_axis(1, false) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionLargerError(1, 0));
            }
        }
    }

    #[test]
    fn argmax_with_axis_normal() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [2, 3, 2]).unwrap();
        let y = x.argmax_with_axis(0, false).unwrap();
        assert_eq!(y.get_data(), &vec![1, 1, 1, 1, 1, 1]);
        assert_eq!(y.get_shape(), &vec![3, 2]);
        let y = x.argmax_with_axis(0, true).unwrap();
        assert_eq!(y.get_data(), &vec![1, 1, 1, 1, 1, 1]);
        assert_eq!(y.get_shape(), &vec![1, 3, 2]);
        let y = x.argmax_with_axis(1, false).unwrap();
        assert_eq!(y.get_data(), &vec![2, 2, 2, 2]);
        assert_eq!(y.get_shape(), &vec![2, 2]);
        let y = x.argmax_with_axis(1, true).unwrap();
        assert_eq!(y.get_data(), &vec![2, 2, 2, 2]);
        assert_eq!(y.get_shape(), &vec![2, 1, 2]);
        let y = x.argmax_with_axis(2, false).unwrap();
        assert_eq!(y.get_data(), &vec![1, 1, 1, 1, 1, 1]);
        assert_eq!(y.get_shape(), &vec![2, 3]);
        let y = x.argmax_with_axis(2, true).unwrap();
        assert_eq!(y.get_data(), &vec![1, 1, 1, 1, 1, 1]);
        assert_eq!(y.get_shape(), &vec![2, 3, 1]);
    }

    #[test]
    fn argmax_with_axis_error_empty() {
        let x = Tensor::<f32>::new([], [0]).unwrap();
        match x.argmax_with_axis(0, false) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::EmptyTensorError());
            }
        }
    }

    #[test]
    fn argmax_with_axis_error_axis() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        match x.argmax_with_axis(1, false) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionLargerError(1, 0));
            }
        }
    }

    #[test]
    fn maximum_normal() {
        let x = Tensor::new([0.0, 3.0, 4.0], [3,]).unwrap();
        let y = Tensor::new([1.0, 2.0, 4.0], [3,]).unwrap();
        let z = x.maximum(&y).unwrap();
        assert_eq!(z.get_data(), &vec![1.0, 3.0, 4.0]);
        assert_eq!(z.get_shape(), &vec![3]);
        let x = Tensor::new([0.0, 3.0, 4.0], [3, 1]).unwrap();
        let y = Tensor::new([1.0, 2.0, 4.0], [3, 1]).unwrap();
        let z = x.maximum(&y).unwrap();
        assert_eq!(z.get_data(), &vec![1.0, 3.0, 4.0]);
        assert_eq!(z.get_shape(), &vec![3, 1]);
    }

    #[test]
    fn maximum_empty() {
        let x = Tensor::<f32>::new([], [0]).unwrap();
        let y = Tensor::<f32>::new([], [0]).unwrap();
        let z = x.maximum(&y).unwrap();
        assert_eq!(z.get_data(), &vec![]);
        assert_eq!(z.get_shape(), &vec![0]);
    }

    #[test]
    fn maximum_error_shape() {
        let x = Tensor::new([0.0, 3.0, 4.0], [3,]).unwrap();
        let y = Tensor::new([1.0, 2.0], [2,]).unwrap();
        match x.maximum(&y) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeMismatchError(vec![3], vec![2]));
            }
        }
    }

    #[test]
    fn min_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        assert_eq!(x.min().unwrap(), 0.0);
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1]).unwrap();
        assert_eq!(x.min().unwrap(), 0.0);
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1, 1]).unwrap();
        assert_eq!(x.min().unwrap(), 0.0);
    }

    #[test]
    fn min_empty() {
        let x = Tensor::<f32>::new([], [0]).unwrap();
        match x.min() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::EmptyTensorError());
            }
        }
    }

    #[test]
    fn create_binyary_mask_from_condition_normal() {
        let x = Tensor::new([0.0, 1.0, 2.0, 3.0], [2, 2]).unwrap();
        let y = x.create_binyary_mask_from_condition(|x| *x > 1.0).unwrap();
        assert_eq!(y.get_data(), &vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(y.get_shape(), &vec![2, 2]);
    }
}
