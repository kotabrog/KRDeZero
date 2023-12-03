use anyhow::Result;
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: Clone,
{
    /// Slice the Tensor with the given first dimension index
    ///
    /// # Arguments
    /// 
    /// * `index` - The index of the first dimension
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - The sliced Tensor
    /// 
    /// # Note
    /// 
    /// If tensor is 0 dimensional, `TensorError::DimensionSmallerError` is returned.
    /// If the index is out of range, `TensorError::IndexError` is returned.
    pub fn slice_with_one_index(&self, index: usize) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(TensorError::DimensionSmallerError(0, 1).into());
        }
        if index >= self.shape[0] {
            return Err(TensorError::IndexError(self.shape.clone(), vec![index]).into());
        }
        let mut new_shape = self.shape.clone();
        new_shape.remove(0);
        let size = new_shape.iter().product();
        let mut data = Vec::with_capacity(size);
        data.extend_from_slice(&self.data[index * size..(index + 1) * size]);
        Ok(Tensor::new(data, new_shape)?)
    }

    /// Slice the Tensor with the given first dimension indexes
    /// 
    /// # Arguments
    /// 
    /// * `indexes` - The indexes of the first dimension
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - The sliced Tensor
    /// 
    /// # Note
    /// 
    /// If tensor is 0 dimensional, `TensorError::DimensionSmallerError` is returned.
    /// If the indexes are out of range, `TensorError::IndexError` is returned.
    /// If the indexes are empty, `TensorError::InvalidArgumentError` is returned.
    pub fn slice_with_one_indexes(&self, indexes: &[usize]) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(TensorError::DimensionSmallerError(0, 1).into());
        }
        if indexes.len() == 0 {
            return Err(TensorError::InvalidArgumentError("indexes is empty".to_string()).into());
        }
        let mut new_shape = self.shape.clone();
        new_shape[0] = indexes.len();
        let size: usize = new_shape[1..].iter().product();
        let mut data = Vec::with_capacity(indexes.len() * size);
        for &index in indexes {
            if index >= self.shape[0] {
                return Err(TensorError::IndexError(self.shape.clone(), vec![index]).into());
            }
            data.extend_from_slice(&self.data[index * size..(index + 1) * size]);
        }
        Ok(Tensor::new(data, new_shape)?)
    }

    /// Slice the Tensor with the given indexes
    /// 
    /// # Arguments
    /// 
    /// * `indexes` - The indexes of the each dimension
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - The sliced Tensor
    /// 
    /// # Note
    /// 
    /// If tensor is 0 dimensional, `TensorError::DimensionSmallerError` is returned.
    /// If tensor is not match the indexes dimension, `TensorError::DimensionLargerError` is returned.
    /// If two indexes have different length, `TensorError::DimensionError` is returned.
    /// If the indexes are out of range, `TensorError::IndexError` is returned.
    /// If the indexes are empty, `TensorError::InvalidArgumentError` is returned.
    pub fn slice_with_indexes(&self, indexes: Vec<Vec<usize>>) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(TensorError::DimensionSmallerError(0, 1).into());
        }
        if self.ndim() < indexes.len() {
            return Err(TensorError::DimensionLargerError(indexes.len(), self.ndim()).into());
        }
        if indexes.len() == 0 {
            return Err(TensorError::InvalidArgumentError("indexes is empty".to_string()).into());
        }
        let len = indexes[0].len();
        for index in &indexes {
            if index.len() != len {
                return Err(TensorError::DimensionError(index.len(), len).into());
            }
        }
        let index_num = indexes.len();
        let mut new_shape = self.shape.clone();
        new_shape.drain(0..index_num);
        new_shape.insert(0, len);
        let size: usize = new_shape[1..].iter().product();
        let mut data = Vec::with_capacity(indexes.len() * size);
        let zip_indexes = (0..len)
            .map(|i| indexes.iter().map(|v| v[i])
            .collect::<Vec<usize>>());
        for mut index in zip_indexes {
            for i in 0..index_num {
                if index[i] >= self.shape[i] {
                    return Err(TensorError::IndexError(self.shape.clone(), index).into());
                }
            }
            index.resize(self.ndim(), 0);
            let data_index = self.calc_data_index(index)?;
            data.extend_from_slice(&self.data[data_index..(data_index + size)]);
        }
        Tensor::new(data, new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_with_one_index_normal() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        let y = x.slice_with_one_index(1).unwrap();
        assert_eq!(y.get_data(), &vec![2.0, 3.0]);
        assert_eq!(y.get_shape(), &vec![2]);
    }

    #[test]
    fn slice_with_one_index_one_dimensional() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_one_index(1).unwrap();
        assert_eq!(y.get_data(), &vec![1.0]);
        assert_eq!(y.get_shape(), &vec![]);
    }

    #[test]
    fn slice_with_one_index_scalar_error() {
        let x = Tensor::new([0.0], []).unwrap();
        let y = x.slice_with_one_index(0);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionSmallerError(0, 1));
            }
        }
    }

    #[test]
    fn slice_with_one_index_index_error() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_one_index(3);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![3], vec![3]));
            }
        }
    }

    #[test]
    fn slice_with_one_indexes_normal() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        let y = x.slice_with_one_indexes(&[0, 2]).unwrap();
        assert_eq!(y.get_data(), &vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(y.get_shape(), &vec![2, 2]);
    }

    #[test]
    fn slice_with_one_indexes_one_dimensional() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_one_indexes(&[0, 2]).unwrap();
        assert_eq!(y.get_data(), &vec![0.0, 2.0]);
        assert_eq!(y.get_shape(), &vec![2]);
    }

    #[test]
    fn slice_with_one_indexes_duplicate() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        let y = x.slice_with_one_indexes(&[0, 1, 1, 0]).unwrap();
        assert_eq!(y.get_data(), &vec![0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 0.0, 1.0]);
        assert_eq!(y.get_shape(), &vec![4, 2]);
    }

    #[test]
    fn slice_with_one_indexes_scalar_error() {
        let x = Tensor::new([0.0], []).unwrap();
        let y = x.slice_with_one_indexes(&[0]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionSmallerError(0, 1));
            }
        }
    }

    #[test]
    fn slice_with_one_indexes_index_error() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_one_indexes(&[0, 3]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![3], vec![3]));
            }
        }
    }

    #[test]
    fn slice_with_one_indexes_empty_error() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_one_indexes(&[]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(
                    e,
                    TensorError::InvalidArgumentError("indexes is empty".to_string())
                );
            }
        }
    }

    #[test]
    fn slice_with_indexes_normal() {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        let y = x.slice_with_indexes(vec![vec![0, 2], vec![1, 0]]).unwrap();
        assert_eq!(y.get_data(), &vec![1.0, 4.0]);
        assert_eq!(y.get_shape(), &vec![2]);
    }

    #[test]
    fn slice_with_indexes_one_dimensional() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_indexes(vec![vec![0, 2]]).unwrap();
        assert_eq!(y.get_data(), &vec![0.0, 2.0]);
        assert_eq!(y.get_shape(), &vec![2]);
    }

    #[test]
    fn slice_with_indexes_three_dimensional() {
        let x = Tensor::new(
            vec![
                0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0
            ],
            vec![2, 2, 2]
        ).unwrap();
        let y = x.slice_with_indexes(vec![vec![0, 0, 1], vec![0, 1, 0], vec![0, 0, 1]]).unwrap();
        assert_eq!(y.get_data(), &vec![0.0, 2.0, 5.0]);
        assert_eq!(y.get_shape(), &vec![3]);
    }

    #[test]
    fn slice_with_indexes_low_dimensional() {
        let x = Tensor::new(
            vec![
                0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0
            ],
            vec![2, 2, 2]
        ).unwrap();
        let y = x.slice_with_indexes(vec![vec![0, 1], vec![0, 0]]).unwrap();
        assert_eq!(y.get_data(), &vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(y.get_shape(), &vec![2, 2]);
    }

    #[test]
    fn slice_with_indexes_zero_dimensional_error() {
        let x = Tensor::new([0.0], []).unwrap();
        let y = x.slice_with_indexes(vec![vec![0]]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionSmallerError(0, 1));
            }
        }
    }

    #[test]
    fn slice_with_indexes_dimension_error() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1]).unwrap();
        let y = x.slice_with_indexes(vec![vec![0, 1], vec![0]]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionError(1, 2));
            }
        }
    }

    #[test]
    fn slice_with_indexes_index_error() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_indexes(vec![vec![0, 3]]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(
                    e,
                    TensorError::IndexError(vec![3], vec![3])
                );
            }
        }
    }

    #[test]
    fn slice_with_indexes_empty_error() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_indexes(vec![]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(
                    e,
                    TensorError::InvalidArgumentError("indexes is empty".to_string())
                );
            }
        }
    }

    #[test]
    fn slice_with_indexes_dimension_larger_error() {
        let x = Tensor::new([0.0, 1.0, 2.0], [3,]).unwrap();
        let y = x.slice_with_indexes(vec![vec![0, 1], vec![0, 1], vec![0, 1]]);
        match y {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(
                    e,
                    TensorError::DimensionLargerError(3, 1)
                );
            }
        }
    }
}
