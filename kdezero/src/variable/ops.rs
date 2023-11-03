use super::Variable;

impl std::ops::Add for Variable {
    type Output = Variable;

    fn add(self, rhs: Self) -> Self::Output {
        crate::function::add(&self, &rhs).unwrap()
    }
}

impl std::ops::Add for &Variable {
    type Output = Variable;

    fn add(self, rhs: Self) -> Self::Output {
        crate::function::add(self, rhs).unwrap()
    }
}

impl std::ops::Mul for Variable {
    type Output = Variable;

    fn mul(self, rhs: Self) -> Self::Output {
        crate::function::mul(&self, &rhs).unwrap()
    }
}

impl std::ops::Mul for &Variable {
    type Output = Variable;

    fn mul(self, rhs: Self) -> Self::Output {
        crate::function::mul(self, rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = x0 + x1;
        assert_eq!(*y.data(), 5.0.into());
    }

    #[test]
    #[should_panic]
    fn error_add_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = x0 + x1;
    }

    #[test]
    fn add_ref_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = &x0 + &x1;
        assert_eq!(*y.data(), 5.0.into());
    }

    #[test]
    #[should_panic]
    fn error_add_ref_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = &x0 + &x1;
    }

    #[test]
    fn mul_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = x0 * x1;
        assert_eq!(*y.data(), 6.0.into());
    }

    #[test]
    #[should_panic]
    fn error_mul_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = x0 * x1;
    }

    #[test]
    fn mul_ref_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = &x0 * &x1;
        assert_eq!(*y.data(), 6.0.into());
    }

    #[test]
    #[should_panic]
    fn error_mul_ref_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = &x0 * &x1;
    }
}
