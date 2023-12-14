use anyhow::Result;
use super::DataSet;

use ktensor::{Tensor, tensor::TensorRng};

pub struct DataLoader<T, U>
where
    T: Clone,
    U: Clone,
{
    data_set: Box<dyn DataSet<T, U>>,
    batch_size: usize,
    data_size: usize,
    shuffle: Option<TensorRng>,
    max_iter: usize,
    iter: usize,
    index: Vec<usize>,
}

pub struct BatchIterator<'a, T, U>
where
    T: Clone,
    U: Clone,
{
    data_loader: &'a mut DataLoader<T, U>,
}

impl<T, U> DataLoader<T, U>
where
    T: Clone,
    U: Clone,
{
    pub fn new(
        data_set: Box<dyn DataSet<T, U>>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self> {
        let data_size = data_set.len()?;
        let max_iter = (data_size - 1) / batch_size + 1;
        let mut shuffle = if shuffle {
            Some(TensorRng::new())
        } else {
            None
        };
        let index = Self::reset_index(&mut shuffle, data_size);
        Ok(DataLoader {
            data_set,
            data_size,
            batch_size,
            shuffle,
            max_iter,
            iter: 0,
            index,
        })
    }

    fn reset_index(shuffle: &mut Option<TensorRng>, len: usize) -> Vec<usize> {
        if let Some(shuffle) = shuffle {
            shuffle.permutation(len)
        } else {
            (0..len).collect()
        }
    }

    pub fn reset(&mut self) {
        self.iter = 0;
        self.index = Self::reset_index(&mut self.shuffle, self.data_size);
    }

    pub fn iter(&mut self) -> BatchIterator<T, U> {
        BatchIterator { data_loader: self }
    }

    pub fn len(&self) -> usize {
        self.data_size
    }
}

impl<'a, T, U> Iterator for BatchIterator<'a, T, U>
where
    T: Clone,
    U: Clone,
{
    type Item = Result<(Tensor<T>, Option<Tensor<U>>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.data_loader.iter;
        if i >= self.data_loader.max_iter {
            self.data_loader.reset();
            return None;
        }
        let batch_size = self.data_loader.batch_size;
        let data_size = self.data_loader.data_size;
        let start = i * batch_size;
        let end = std::cmp::min(start + batch_size, data_size);
        let index = self.data_loader.index[start..end]
            .iter()
            .map(|&x| x)
            .collect::<Vec<_>>();
        let mut batch_x = vec![];
        let mut batch_t = vec![];
        let mut label_flag = true;
        for i in index {
            let result = self.data_loader.data_set.get(i);
            let (x, t) = if let Ok((x, t)) = result {
                (x, t)
            } else {
                return Some(result);
            };
            batch_x.push(x);
            if t.is_none() {
                label_flag = false;
            } else {
                batch_t.push(t.unwrap());
            }
        }
        let batch_x = match Tensor::from_tensor_list(&batch_x
            .iter()
            .map(|x| x)
            .collect::<Vec<_>>()) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
        };
        let batch_t = if label_flag {
            match Tensor::from_tensor_list(&batch_t
                .iter()
                .map(|t| t)
                .collect::<Vec<_>>()) {
                    Ok(t) => Some(t),
                    Err(e) => return Some(Err(e)),
            }
        } else {
            None
        };
        self.data_loader.iter += 1;
        Some(Ok((batch_x, batch_t)))
    }
}

impl<'a, T: 'a> IntoIterator for &'a mut DataLoader<T, T>
where
    T: Clone,
{
    type Item = Result<(Tensor<T>, Option<Tensor<T>>)>;
    type IntoIter = BatchIterator<'a, T, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
