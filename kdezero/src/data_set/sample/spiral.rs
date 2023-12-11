use anyhow::Result;
use ktensor::{Tensor, tensor::TensorRng};
use super::super::DataSet;

pub fn get_spiral(train: bool) -> Result<(Tensor<f64>, Tensor<usize>)> {
    let seed = if train { 1984 } else { 2020 };
    let mut rng = TensorRng::new_from_seed(seed);

    let num_data = 100;
    let num_class = 3;
    let input_dim = 2;
    let data_size = num_data * num_class;
    let mut x = Tensor::<f64>::zeros([data_size, input_dim]);
    let mut t = Tensor::<usize>::zeros([data_size]);

    for j in 0..num_class {
        for i in 0..num_data {
            let rate = i as f64 / num_data as f64;
            let radius = 1.0 * rate;
            let random = rng.standard::<f64, _>([])?
                .to_scalar()?;
            let theta =
                j as f64 * 4.0 + 4.0 * rate
                + random * 0.2;
            let ix = num_data * j + i;
            let x0 = x.at_mut([ix, 0])?;
            *x0 = radius * theta.sin();
            let x1 = x.at_mut([ix, 1])?;
            *x1 = radius * theta.cos();
            let t = t.at_mut([ix])?;
            *t = j;
        }
    }
    let indices = rng.permutation(data_size);
    let x = x.slice_with_one_indexes(&indices)?;
    let t = t.slice_with_one_indexes(&indices)?;
    Ok((x, t))
}

pub struct Spiral {
    pub train: bool,
    pub data: Tensor<f64>,
    pub label: Tensor<usize>,
}

impl Spiral {
    pub fn new(train: bool) -> Result<Self> {
        let (x, t) = get_spiral(train)?;
        Ok(Self {
            train,
            data: x,
            label: t,
        })
    }
}

impl DataSet<f64, usize> for Spiral {
    fn get_all_data(&self) -> Result<Option<&Tensor<f64>>> {
        Ok(Some(&self.data))
    }

    fn get_all_label(&self) -> Result<Option<&Tensor<usize>>> {
        Ok(Some(&self.label))
    }
}
