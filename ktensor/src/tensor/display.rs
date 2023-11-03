use super::Tensor;

const INDENT: usize = 7;

fn make_switch_timing(shape: &Vec<usize>) -> Vec<usize> {
    let mut shape = shape.clone();
    shape.reverse();
    let mut switch_timing = Vec::new();
    let mut count = 1;
    for i in 0..shape.len() - 1 {
        count *= shape[i];
        switch_timing.push(count);
    }
    switch_timing
}

fn add_start_bracket(s: &mut String, switch_timing: &Vec<usize>,
                     index: usize, bracket: &mut usize) {
    let mut count = 0;
    for i in 0..switch_timing.len() {
        if index % switch_timing[i] == 0 {
            count += 1;
        }
    }
    if count > 0 {
        s.push_str(&format!("{}", "[".repeat(count)));
        *bracket += count;
    }
}

fn add_end_bracket(s: &mut String, switch_timing: &Vec<usize>,
                   index: usize, bracket: &mut usize) {
    let mut count = 0;
    let mut newline = 0;
    for i in 0..switch_timing.len() {
        if (index + 1) % switch_timing[i] == 0 {
            count += 1;
            if i > 0 {
                newline = newline.max(2);
            } else {
                newline = newline.max(1);
            }
        }
    }
    if count > 0 {
        *bracket -= count;
        s.push_str(&format!("{},{}{}",
            "]".repeat(count), "\n".repeat(newline), " ".repeat(INDENT + *bracket)));
    } else {
        s.push_str(", ");
    }
}

impl<T> std::fmt::Display for Tensor<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        if self.shape.len() == 0 {
            s.push_str(&format!("tensor({}, type={})", self.data[0], self.data_type()));
            return write!(f, "{}", s);
        }
        if self.data.len() == 0 {
            s.push_str(&format!("tensor(shape={:?}, type={})", self.get_shape() as &[usize], self.data_type()));
            return write!(f, "{}", s);
        }
        let switch_timing = make_switch_timing(&self.shape);
        s.push_str("tensor([");
        let mut bracket = 1;
        for i in 0..self.data.len() {
            add_start_bracket(&mut s, &switch_timing, i, &mut bracket);

            s.push_str(&format!("{}", self.data[i]));

            if i == self.data.len() - 1 {
                break;
            }
            add_end_bracket(&mut s, &switch_timing, i, &mut bracket);
        }
        s.push_str(&format!("{}, type={})", "]".repeat(bracket), self.data_type()));
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_scalar() {
        let x = Tensor::new([1], [1]).unwrap();
        assert_eq!(format!("{}", x), "tensor([1], type=i32)");
    }

    #[test]
    fn display_vector() {
        let x = Tensor::new([0, 1, 2], [3]).unwrap();
        assert_eq!(format!("{}", x), "tensor([0, 1, 2], type=i32)");
    }

    #[test]
    fn display_matrix() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5], [2, 3]).unwrap();
        assert_eq!(format!("{}", x), "tensor([[0, 1, 2],\n        [3, 4, 5]], type=i32)");
    }

    #[test]
    fn display_3_dim() {
        let x = Tensor::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2, 3, 2]).unwrap();
        assert_eq!(format!("{}", x), "tensor([[[0, 1],\n         [2, 3],\n         [4, 5]],\n\n        [[6, 7],\n         [8, 9],\n         [10, 11]]], type=i32)");
    }

    #[test]
    fn display_zero_dim() {
        let x = Tensor::new([1], []).unwrap();
        assert_eq!(format!("{}", x), "tensor(1, type=i32)");
    }

    #[test]
    fn display_zero_shape() {
        let x: Tensor<f64> = Tensor::new([], [2, 0]).unwrap();
        assert_eq!(format!("{}", x), "tensor(shape=[2, 0], type=f64)");
    }

    #[test]
    fn display_zero_shape_2() {
        let x: Tensor<f64> = Tensor::new([], [0, 2]).unwrap();
        assert_eq!(format!("{}", x), "tensor(shape=[0, 2], type=f64)");
    }
}
