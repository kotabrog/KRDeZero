use anyhow::Result;
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: Clone
{
    fn ops_at_one_index(&self, rhs: &Self, f: fn (T, T) -> T, index: usize) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(TensorError::DimensionSmallerError(0, 1).into())
        }
        if index >= self.shape[0] {
            return Err(TensorError::IndexError(self.shape.clone(), vec![index]).into())
        }
        if self.ndim() - 1 != rhs.ndim() {
            return Err(TensorError::DimensionError(rhs.ndim(), self.ndim() - 1).into())
        }
        for i in 0..rhs.ndim() {
            if self.shape[i + 1] != rhs.shape[i] {
                return Err(TensorError::ShapeError(self.shape.clone(), rhs.shape.clone()).into())
            }
        }
        let shape = self.shape.clone();
        let size = rhs.size();
        let mut data = self.data.clone();
        for i in 0..size {
            data[index * size + i] = f(data[index * size + i].clone(), rhs.data[i].clone());
        }
        Tensor::new(data, shape)
    }

    fn ops_at_one_indexes(&self, rhs: &Self, f: fn (T, T) -> T, indexes: &[usize]) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(TensorError::DimensionSmallerError(0, 1).into())
        }
        if self.ndim() != rhs.ndim() {
            return Err(TensorError::DimensionError(rhs.ndim(), self.ndim()).into())
        }
        if rhs.shape[0] != indexes.len() {
            return Err(TensorError::DimensionError(indexes.len(), rhs.shape[0]).into())
        }
        for &index in indexes {
            if index >= self.shape[0] {
                return Err(TensorError::IndexError(self.shape.clone(), vec![index]).into())
            }
        }
        for i in 1..rhs.ndim() {
            if self.shape[i] != rhs.shape[i] {
                return Err(TensorError::ShapeError(self.shape.clone(), rhs.shape.clone()).into())
            }
        }
        let shape = self.shape.clone();
        let size = rhs.size() / rhs.shape[0];
        let mut data = self.data.clone();
        for (i, index) in indexes.iter().enumerate() {
            for j in 0..size {
                data[index * size + j] = f(
                    data[index * size + j].clone(),
                    rhs.data[i * size + j].clone());
            }
        }
        Tensor::new(data, shape)
    }

    fn ops_at_with_indexes(&self, rhs: &Self, f: fn (T, T) -> T, indexes: Vec<Vec<usize>>) -> Result<Self> {
        if self.ndim() == 0 {
            return Err(TensorError::DimensionSmallerError(0, 1).into())
        }
        if rhs.ndim() == 0 {
            return Err(TensorError::DimensionSmallerError(0, 1).into())
        }
        let index_len = indexes.len();
        if rhs.ndim() == 1 && self.ndim() != 1 {
            if self.ndim() != index_len + 1 {
                return Err(TensorError::DimensionError(index_len + 1, self.ndim()).into())
            }
        } else {
            if self.ndim() != rhs.ndim() + index_len - 1 {
                return Err(TensorError::DimensionError(rhs.ndim() + index_len - 1, self.ndim()).into())
            }
        }
        for (i, index) in indexes.iter().enumerate() {
            for &j in index {
                if j >= self.shape[i] {
                    return Err(TensorError::IndexError(self.shape.clone(), vec![j]).into())
                }
            }
        }
        if rhs.ndim() == 1 && self.shape[index_len - 1] != 1 {
            return Err(TensorError::ShapeError(self.shape.clone(), rhs.shape.clone()).into())
        }
        for i in 1..rhs.ndim() {
            if self.shape[i + index_len - 1] != rhs.shape[i] {
                return Err(TensorError::ShapeError(self.shape.clone(), rhs.shape.clone()).into())
            }
        }
        let len = indexes[0].len();
        if rhs.shape[0] != len {
            return Err(TensorError::DimensionError(rhs.shape[0], len).into())
        }
        for index in &indexes {
            if index.len() != len {
                return Err(TensorError::DimensionError(index.len(), len).into())
            }
        }
        let shape = self.shape.clone();
        let size = rhs.size() / rhs.shape[0];
        let mut data = self.data.clone();
        let zip_indexes = (0..len)
            .map(|i| indexes.iter().map(|v| v[i])
            .collect::<Vec<_>>());
        for (i, mut index) in zip_indexes.enumerate() {
            index.resize(self.ndim(), 0);
            let data_index = self.calc_data_index(index)?;
            for j in 0..size {
                data[data_index + j] = f(
                    data[data_index + j].clone(),
                    rhs.data[i * size + j].clone()
                );
            }
        }
        Tensor::new(data, shape)
    }
}

impl<T> Tensor<T>
where
    T: std::ops::Add<Output = T> + Clone
{
    /// Add a tensor at first dimension index
    /// 
    /// # Arguments
    /// 
    /// * `rhs` - The tensor to add
    /// * `index` - The index to add
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the addition
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, `TensorError::ShapeError` is returned
    /// If the index is out of range, `TensorError::IndexError` is returned
    /// If the dimension is not correct, `TensorError::DimensionError` is returned
    /// If tensor is 0 dimension, `TensorError::DimensionSmallerError` is returned
    pub fn add_at_one_index(&self, rhs: &Self, index: usize) -> Result<Self> {
        self.ops_at_one_index(rhs, |x, y| x + y, index)
    }

    /// Add a tensor at first dimension indexes
    /// 
    /// # Arguments
    /// 
    /// * `rhs` - The tensor to add
    /// * `indexes` - The indexes to add
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the addition
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, `TensorError::ShapeError` is returned
    /// If the index is out of range, `TensorError::IndexError` is returned
    /// If the dimension is not correct, `TensorError::DimensionError` is returned
    /// If tensor is 0 dimension, `TensorError::DimensionSmallerError` is returned
    pub fn add_at_one_indexes(&self, rhs: &Self, indexes: &[usize]) -> Result<Self> {
        self.ops_at_one_indexes(rhs, |x, y| x + y, indexes)
    }

    /// Add a tensor at indexes
    /// 
    /// # Arguments
    /// 
    /// * `rhs` - The tensor to add
    /// * `indexes` - The indexes to add
    /// 
    /// # Returns
    /// 
    /// * `Result<Self>` - Result of the addition
    /// 
    /// # Note
    /// 
    /// If the shape is not correct, `TensorError::ShapeError` is returned
    /// If the index is out of range, `TensorError::IndexError` is returned
    /// If the dimension is not correct, `TensorError::DimensionError` is returned
    /// If tensor or indexes is 0 dimension, `TensorError::DimensionSmallerError` is returned
    pub fn add_at_with_indexes(&self, rhs: &Self, indexes: Vec<Vec<usize>>) -> Result<Self> {
        self.ops_at_with_indexes(rhs, |x, y| x + y, indexes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_at_one_index() -> Result<()> {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        ).unwrap();
        let y = Tensor::new([1.0, 2.0], [2])?;
        let z = x.add_at_one_index(&y, 2)?;
        assert_eq!(z, Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 5.0, 7.0],
            [3, 2]
        )?);
        Ok(())
    }

    #[test]
    fn add_at_one_index_one_dimensional() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3])?;
        let y = Tensor::new([2.0], [])?;
        let z = x.add_at_one_index(&y, 1)?;
        assert_eq!(z, Tensor::new([0.0, 3.0, 2.0], [3])?);
        Ok(())
    }

    #[test]
    fn add_at_one_index_error_zero_dimension() -> Result<()> {
        let x = Tensor::new([0.0], [])?;
        let y = Tensor::new([2.0], [])?;
        let z = x.add_at_one_index(&y, 1);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionSmallerError(0, 1));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_one_index_error_index() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3])?;
        let y = Tensor::new([2.0], [])?;
        let z = x.add_at_one_index(&y, 3);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![3], vec![3]));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_one_index_error_not_match_dimension() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3])?;
        let y = Tensor::new([2.0], [1])?;
        let z = x.add_at_one_index(&y, 1);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionError(1, 0));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_one_index_error_not_match_shape() -> Result<()> {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        )?;
        let y = Tensor::new([2.0, 3.0, 4.0], [3])?;
        let z = x.add_at_one_index(&y, 1);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![3, 2], vec![3]));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_one_indexes() -> Result<()> {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        )?;
        let y = Tensor::new(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2]
        )?;
        let z = x.add_at_one_indexes(&y, &[1, 2])?;
        assert_eq!(z, Tensor::new(
            [0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
            [3, 2]
        )?);
        Ok(())
    }

    #[test]
    fn add_at_one_indexes_error_zero_dimension() -> Result<()> {
        let x = Tensor::new([0.0], [])?;
        let y = Tensor::new([2.0], [])?;
        let z = x.add_at_one_indexes(&y, &[1, 2]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionSmallerError(0, 1));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_one_indexes_error_index() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3])?;
        let y = Tensor::new([2.0], [1])?;
        let z = x.add_at_one_indexes(&y, &[3]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::IndexError(vec![3], vec![3]));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_one_indexes_error_not_match_dimension() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3])?;
        let y = Tensor::new([2.0], [])?;
        let z = x.add_at_one_indexes(&y, &[1, 2]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionError(0, 1));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_one_indexes_error_not_match_shape() -> Result<()> {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        )?;
        let y = Tensor::new([2.0, 3.0, 4.0], [3, 1])?;
        let z = x.add_at_one_indexes(&y, &[1, 2, 2]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![3, 2], vec![3, 1]));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_with_indexes() -> Result<()> {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [2, 2, 2]
        )?;
        let y = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])?;
        let z = x.add_at_with_indexes(&y, vec![vec![1, 0], vec![0, 1]])?;
        assert_eq!(z, Tensor::new(
            [0.0, 1.0, 5.0, 7.0, 5.0, 7.0, 6.0, 7.0],
            [2, 2, 2]
        )?);
        Ok(())
    }

    #[test]
    fn add_at_with_indexes_error_zero_dimension() -> Result<()> {
        let x = Tensor::new([0.0], [])?;
        let y = Tensor::new([2.0], [])?;
        let z = x.add_at_with_indexes(&y, vec![vec![1, 2]]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionSmallerError(0, 1));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_with_indexes_error_not_match_dimension() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3])?;
        let y = Tensor::new([2.0, 3.0], [2])?;
        let z = x.add_at_with_indexes(&y, vec![vec![1, 2], vec![1, 2]]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::DimensionError(2, 1));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_with_indexes_error_not_match_shape() -> Result<()> {
        let x = Tensor::new(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [3, 2]
        )?;
        let y = Tensor::new([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 3])?;
        let z = x.add_at_with_indexes(&y, vec![vec![1, 0]]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(e, TensorError::ShapeError(vec![3, 2], vec![2, 3]));
            }
        }
        Ok(())
    }

    #[test]
    fn add_at_with_indexes_error_not_match_indexes_dimension() -> Result<()> {
        let x = Tensor::new([0.0, 1.0, 2.0], [3, 1, 1])?;
        let y = Tensor::new([2.0, 3.0], [2])?;
        let z = x.add_at_with_indexes(&y, vec![vec![1, 2], vec![0]]);
        match z {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().unwrap();
                assert_eq!(
                    e,
                    TensorError::DimensionError(1, 2)
                );
            }
        }
        Ok(())
    }
}
