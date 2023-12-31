use anyhow::Result;
use num_traits::{PrimInt, Float, NumAssign};
use super::Tensor;
use crate::error::TensorError;

impl<T> Tensor<T>
where
    T: PrimInt
{
    /// Calculate the power of the tensor
    /// 
    /// # Arguments
    /// 
    /// * `n` - The power to be calculated (unsigned integer)
    pub fn pow(&self, n: u32) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.pow(n))
                .collect(),
            shape: self.shape.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Float
{
    /// Calculate the power of the tensor
    /// 
    /// # Arguments
    /// 
    /// * `n` - The power to be calculated (integer)
    pub fn powi(&self, n: i32) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.powi(n))
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Calculate the power of the tensor
    /// 
    /// # Arguments
    /// 
    /// * `n` - The power to be calculated (floating point)
    pub fn powf(&self, n: T) -> Self {
        Self {
            data: self.data
                .iter()
                .map(|x| x.powf(n))
                .collect(),
            shape: self.shape.clone(),
        }
    }

    /// Calculate the exponential of the tensor
    pub fn exp(&self) -> Self {
        self.iter_func(|x| x.exp())
    }

    /// Calculate the natural logarithm of the tensor
    pub fn log(&self) -> Self {
        self.iter_func(|x| x.ln())
    }

    /// Calculate the sin of the tensor
    pub fn sin(&self) -> Self {
        self.iter_func(|x| x.sin())
    }

    /// Calculate the cos of the tensor
    pub fn cos(&self) -> Self {
        self.iter_func(|x| x.cos())
    }

    /// Calculate the tanh of the tensor
    pub fn tanh(&self) -> Self {
        self.iter_func(|x| x.tanh())
    }

    /// Calculate the absolute value of the tensor
    pub fn abs(&self) -> Self {
        self.iter_func(|x| x.abs())
    }

    /// Calculate the square root of the tensor
    pub fn sqrt(&self) -> Self {
        self.iter_func(|x| x.sqrt())
    }
}

impl<T> Tensor<T>
where
    T: Float + NumAssign
{
    /// Calculate the mean of the tensor
    pub fn mean(&self) -> Result<T> {
        let len = T::from(self.size())
            .ok_or(TensorError::CastError(
                std::any::type_name::<T>().to_string()
            ))?;
        Ok(self.sum_all() / len)
    }

    /// Calculate the standard deviation of the tensor
    pub fn std(&self) -> Result<T> {
        let len = T::from(self.size())
            .ok_or(TensorError::CastError(
                std::any::type_name::<T>().to_string()
            ))?;
        let mean = self.mean()?;
        let mut sum = T::zero();
        for x in self.data.iter() {
            sum += (*x - mean).powi(2);
        }
        Ok((sum / len).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_test() {
        let x = Tensor::new(vec![0, 1, 2, 3], vec![2, 2]).unwrap();
        assert_eq!(x.pow(2), Tensor::new(vec![0, 1, 4, 9], vec![2, 2]).unwrap());
    }

    #[test]
    fn powi_normal() {
        let x = Tensor::arrange([2, 2]).unwrap();
        assert_eq!(x.powi(2), Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn powf_normal() {
        let x = Tensor::arrange([2, 2]).unwrap();
        assert_eq!(x.powf(2.0), Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn exp_normal() {
        let x = Tensor::<f64>::arrange([2, 2]).unwrap();
        assert_eq!(x.exp(), Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0].iter().map(|x| x.exp()).collect::<Vec<_>>(),
            vec![2, 2]
        ).unwrap());
    }

    #[test]
    fn log_normal() {
        let x = Tensor::<f64>::arrange([2, 2]).unwrap();
        assert_eq!(x.log(), Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0].iter().map(|x| x.ln()).collect::<Vec<_>>(),
            vec![2, 2]
        ).unwrap());
    }

    #[test]
    fn sin_normal() {
        let x = Tensor::<f64>::arrange([2, 2]).unwrap();
        assert_eq!(x.sin(), Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0].iter().map(|x| x.sin()).collect::<Vec<_>>(),
            vec![2, 2]
        ).unwrap());
    }

    #[test]
    fn cos_normal() {
        let x = Tensor::<f64>::arrange([2, 2]).unwrap();
        assert_eq!(x.cos(), Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0].iter().map(|x| x.cos()).collect::<Vec<_>>(),
            vec![2, 2]
        ).unwrap());
    }

    #[test]
    fn tanh_normal() {
        let x = Tensor::<f64>::arrange([2, 2]).unwrap();
        assert_eq!(x.tanh(), Tensor::new(
            vec![0.0, 1.0, 2.0, 3.0].iter().map(|x| x.tanh()).collect::<Vec<_>>(),
            vec![2, 2]
        ).unwrap());
    }

    #[test]
    fn abs_normal() {
        let x = Tensor::<f64>::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(x.abs(), Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn sqrt_normal() {
        let x = Tensor::<f64>::new(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2]).unwrap();
        assert_eq!(x.sqrt(), Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn mean_normal() {
        let x = Tensor::<f64>::arrange([2, 2]).unwrap();
        assert_eq!(x.mean().unwrap(), 1.5);
    }

    #[test]
    fn std_normal() {
        let x = Tensor::<f64>::arrange([2, 2]).unwrap();
        assert!(x.std().unwrap() - 1.118033988749895 < 1e-10);
    }
}
