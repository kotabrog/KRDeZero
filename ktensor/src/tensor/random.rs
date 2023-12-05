use anyhow::Result;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand::distributions::{Distribution, Standard};
use rand_distr::{Normal, StandardNormal};
use num_traits::Float;
use super::Tensor;
use crate::error::TensorError;

/// Random number generator for Tensor.
/// 
/// # Fields
/// 
/// * `rng` - Random number generator.
pub struct TensorRng {
    rng: StdRng,
}

impl TensorRng {
    /// Create a new TensorRng.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a new TensorRng with seed.
    pub fn new_from_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate a random Tensor.
    /// 
    /// # Arguments
    /// 
    /// * `shape` - Shape of the Tensor.
    pub fn gen<T, U>(&mut self, shape: U) -> Tensor<T>
    where
        Standard: Distribution<T>,
        T: Clone + Default,
        U: Into<Vec<usize>>
    {
        let shape = shape.into();
        let size: usize = shape.iter().product();
        let mut data = vec![T::default(); size];
        for i in 0..size {
            data[i] = self.rng.gen::<T>();
        }
        Tensor { data, shape: shape }
    }

    /// Generate a random Tensor with normal distribution.
    /// 
    /// # Arguments
    /// 
    /// * `shape` - Shape of the Tensor.
    /// * `mean` - Mean of the normal distribution.
    /// * `std` - Standard deviation of the normal distribution.
    pub fn normal<T, U>(&mut self, shape: U, mean: T, std: T) -> Result<Tensor<T>>
    where
        StandardNormal: Distribution<T>,
        T: Float + Default,
        U: Into<Vec<usize>>
    {
        let shape = shape.into();
        let size: usize = shape.iter().product();
        let mut data = vec![T::default(); size];
        let normal = Normal::new(mean, std)
            .map_err(|_| TensorError::NewRandomNormalError())?;
        for i in 0..size {
            data[i] = self.rng.sample(normal).into();
        }
        Ok(Tensor { data, shape: shape })
    }

    /// Generate a random Tensor with standard normal distribution.
    /// 
    /// # Arguments
    /// 
    /// * `shape` - Shape of the Tensor.
    pub fn standard<T, U>(&mut self, shape: U) -> Result<Tensor<T>>
    where
        StandardNormal: Distribution<T>,
        T: Float + Default,
        U: Into<Vec<usize>>,
    {
        self.normal(shape, T::zero(), T::one())
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        v.shuffle(&mut self.rng);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_normal() {
        let mut rng = TensorRng::new();
        let x = rng.gen::<f32, _>([2, 3]);
        assert_eq!(x.data_type(), "f32");
        assert_eq!(x.get_shape(), &[2, 3]);
    }

    #[test]
    fn gen_scalar() {
        let mut rng = TensorRng::new();
        let x = rng.gen::<f32, _>([]);
        assert_eq!(x.data_type(), "f32");
        assert_eq!(x.get_shape(), &[]);
    }

    #[test]
    fn new_from_seed() {
        let mut rng = TensorRng::new_from_seed(0);
        let x0 = rng.gen::<f32, _>([2, 3]);
        let mut rng = TensorRng::new_from_seed(0);
        let x1 = rng.gen::<f32, _>([2, 3]);
        assert_eq!(x0, x1);
    }

    #[test]
    fn normal_normal() {
        let mut rng = TensorRng::new();
        let x = rng.normal([2, 3], 0.0f32, 1.0).unwrap();
        assert_eq!(x.data_type(), "f32");
        assert_eq!(x.get_shape(), &[2, 3]);
    }

    #[test]
    fn normal_check_mean_std() {
        let mut rng = TensorRng::new();
        let x = rng.normal([1000], 0.0, 1.0).unwrap();
        let mean = x.mean().unwrap();
        let std = x.std().unwrap();
        assert!((mean - 0.0).abs() < 0.1);
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn standard_normal() {
        let mut rng = TensorRng::new();
        let x = rng.standard::<f64, _>([2, 3]).unwrap();
        assert_eq!(x.data_type(), "f64");
        assert_eq!(x.get_shape(), &[2, 3]);
    }

    #[test]
    fn standard_check_mean_std() {
        let mut rng = TensorRng::new();
        let x = rng.standard::<f64, _>([1000]).unwrap();
        let mean = x.mean().unwrap();
        let std = x.std().unwrap();
        assert!((mean - 0.0).abs() < 0.1);
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn permutation_normal() {
        let mut rng = TensorRng::new();
        let x = rng.permutation(10);
        assert_eq!(x.len(), 10);
        for i in 0..10 {
            assert!(x.contains(&i));
        }
    }
}
