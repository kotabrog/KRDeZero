pub mod sample;

use anyhow::Result;
use ktensor::Tensor;

pub type Transform<T> = fn(&Tensor<T>) -> Result<Tensor<T>>;

pub trait DataSet<T, U>
where
    T: Clone,
    U: Clone,
{
    fn get_all_data(&self) -> Result<Option<&Tensor<T>>> {
        Ok(None)
    }

    fn get_all_label(&self) -> Result<Option<&Tensor<U>>> {
        Ok(None)
    }

    fn len(&self) -> Result<usize> {
        let v = self.get_all_data()?;
        if let Some(v) = v {
            Ok(v.len())
        } else {
            Ok(0)
        }
    }

    fn get_transform(&self) -> Transform<T> {
        |x| Ok(x.clone())
    }

    fn get_target_transform(&self) -> Transform<U> {
        |x| Ok(x.clone())
    }

    fn get_raw_data(&self, index: usize) -> Result<(Tensor<T>, Option<Tensor<U>>)> {
        let x = self.get_all_data()?;
        let t = self.get_all_label()?;
        if let Some(x) = x {
            let x = x.slice_with_one_index(index)?;
            let t = if let Some(t) = t {
                Some(t.slice_with_one_index(index)?)
            } else {
                None
            };
            return Ok((x, t));
        }
        Err(anyhow::anyhow!("Not implemented"))
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Option<Tensor<U>>)> {
        let (x, t) = self.get_raw_data(index)?;
        let x = (self.get_transform())(&x)?;
        if let Some(t) = t {
            let t = (self.get_target_transform())(&t)?;
            return Ok((x, Some(t)));
        } else {
            return Ok((x, None));
        }
    }
}
