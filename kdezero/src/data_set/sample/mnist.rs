use std::env;
use std::io::Read;
use std::fs::{File, create_dir};
use std::path::{Path, PathBuf};
use anyhow::Result;
use flate2::read::GzDecoder;
use ktensor::Tensor;
use super::super::{DataSet, Transform};

fn get_home_dir() -> Result<PathBuf> {
    let home_dir = env::var("HOME")?;
    let path = PathBuf::from(home_dir);
    Ok(path)
}

fn create_default_dir() -> Result<PathBuf> {
    let mut dir_path = get_home_dir()?;
    dir_path.push("data");
    match create_dir(&dir_path) {
        Ok(_) => println!("Create data directory"),
        Err(_) => {},
    }
    dir_path.push("mnist");
    match create_dir(&dir_path) {
        Ok(_) => println!("Create mnist directory"),
        Err(_) => {},
    }
    Ok(dir_path)
}

fn load_mnist(dir_path: Option<&str>) -> Result<()> {
    let url = "http://yann.lecun.com/exdb/mnist/";
    let dir_path = if let Some(dir_path) = dir_path {
        PathBuf::from(dir_path)
    } else {
        create_default_dir()?
    };
    let files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ];
    for file in files {
        let dst_filename = file.split('.').next().unwrap();
        let mut path = dir_path.clone();
        path.push(dst_filename);
        let path = Path::new(&path);
        if path.exists() {
            if let Some(s) = path.to_str() {
                println!("{} already exists", s);
            }
            continue;
        }
        let url = format!("{}{}", url, file);
        let body = reqwest::blocking::get(&url)?.bytes()?;
        let mut gz = GzDecoder::new(body.as_ref());
        let mut dst = File::create(path)?;
        std::io::copy(&mut gz, &mut dst)?;
    }
    Ok(())
}

fn mnist_image_to_vec(path: &Path) -> Result<Vec<Vec<u8>>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut images = Vec::new();
    let offset = 16;
    let num_images = (buffer.len() - offset) / (28 * 28);
    for i in 0..num_images {
        let start = offset + i * 28 * 28;
        let image_data = buffer[start..start + 28 * 28].to_vec();
        images.push(image_data);
    }
    Ok(images)
}

fn mnist_label_to_vec(path: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let offset = 8;
    let labels = buffer[offset..].to_vec();
    Ok(labels)
}

pub struct Mnist {
    pub train: bool,
    pub data: Vec<Vec<u8>>,
    pub label: Vec<u8>,
    pub transform: Transform<f32>,
}

impl Mnist {
    pub fn new(train: bool, dir_path: Option<&str>, transform: Transform<f32>) -> Result<Self> {
        load_mnist(dir_path)?;
        let path = if train {
            PathBuf::from("output/mnist/train-images-idx3-ubyte")
        } else {
            PathBuf::from("output/mnist/t10k-images-idx3-ubyte")
        };
        let data = mnist_image_to_vec(path.as_path())?;
        let path = if train {
            PathBuf::from("output/mnist/train-labels-idx1-ubyte")
        } else {
            PathBuf::from("output/mnist/t10k-labels-idx1-ubyte")
        };
        let label = mnist_label_to_vec(path.as_path())?;
        // let labels: Vec<usize> = labels
        //     .iter()
        //     .map(|x| *x as usize)
        //     .collect();
        // let images: Vec<f32> = images
        //     .iter()
        //     .flat_map(|x| x.iter())
        //     .map(|x| *x as f32)
        //     .collect();
        // let data_len = images.len() / (28 * 28);
        // let data = Tensor::new(
        //     images,
        //     [data_len, 28, 28])?;
        // let label = Tensor::new(labels, [data_len])?;
        Ok(Self {
            train,
            data,
            label,
            transform,
        })
    }
}

impl DataSet<f32, usize> for Mnist {
    fn get_all_data(&self) -> Result<Option<&Tensor<f32>>> {
        Ok(None)
    }

    fn get_all_label(&self) -> Result<Option<&Tensor<usize>>> {
        Ok(None)
    }

    fn len(&self) -> Result<usize> {
        Ok(self.data.len())
    }

    fn get_transform(&self) -> Transform<f32> {
        self.transform
    }

    fn get_raw_data(&self, index: usize) -> Result<(Tensor<f32>, Option<Tensor<usize>>)> {
        let data = self.data[index].clone();
        let data = Tensor::new(
            data.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            [28, 28])?;
        let label = self.label[index];
        let label = Tensor::scalar(label as usize);
        Ok((data, Some(label)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_mnist() -> Result<()> {
        match create_dir("output") {
            Ok(_) => println!("Create data directory"),
            Err(_) => {},
        }
        match create_dir("output/mnist") {
            Ok(_) => println!("Create mnist directory"),
            Err(_) => {},
        }
        load_mnist(Some("output/mnist"))?;
        Ok(())
    }

    #[test]
    fn test_get_images() -> Result<()> {
        match create_dir("output") {
            Ok(_) => println!("Create data directory"),
            Err(_) => {},
        }
        match create_dir("output/mnist") {
            Ok(_) => println!("Create mnist directory"),
            Err(_) => {},
        }
        load_mnist(Some("output/mnist"))?;
        let path = PathBuf::from("output/mnist/train-images-idx3-ubyte");
        let images = mnist_image_to_vec(path.as_path())?;
        assert_eq!(images.len(), 60000);
        assert_eq!(images[0].len(), 28 * 28);
        Ok(())
    }

    #[test]
    fn test_get_labels() -> Result<()> {
        match create_dir("output") {
            Ok(_) => println!("Create data directory"),
            Err(_) => {},
        }
        match create_dir("output/mnist") {
            Ok(_) => println!("Create mnist directory"),
            Err(_) => {},
        }
        load_mnist(Some("output/mnist"))?;
        let path = PathBuf::from("output/mnist/train-labels-idx1-ubyte");
        let tensor = mnist_label_to_vec(path.as_path())?;
        assert_eq!(tensor.len(), 60000);
        assert_eq!(tensor[0], 5);
        Ok(())
    }
}
