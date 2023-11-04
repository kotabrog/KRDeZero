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

impl std::ops::Sub for Variable {
    type Output = Variable;

    fn sub(self, rhs: Self) -> Self::Output {
        crate::function::sub(&self, &rhs).unwrap()
    }
}

impl std::ops::Sub for &Variable {
    type Output = Variable;

    fn sub(self, rhs: Self) -> Self::Output {
        crate::function::sub(self, rhs).unwrap()
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

impl std::ops::Div for Variable {
    type Output = Variable;

    fn div(self, rhs: Self) -> Self::Output {
        crate::function::div(&self, &rhs).unwrap()
    }
}

impl std::ops::Div for &Variable {
    type Output = Variable;

    fn div(self, rhs: Self) -> Self::Output {
        crate::function::div(self, rhs).unwrap()
    }
}

impl std::ops::Neg for Variable {
    type Output = Variable;

    fn neg(self) -> Self::Output {
        crate::function::neg(&self).unwrap()
    }
}

impl std::ops::Neg for &Variable {
    type Output = Variable;

    fn neg(self) -> Self::Output {
        crate::function::neg(self).unwrap()
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
    fn sub_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = x0 - x1;
        assert_eq!(*y.data(), (-1.0).into());
    }

    #[test]
    #[should_panic]
    fn error_sub_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = x0 - x1;
    }

    #[test]
    fn sub_ref_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = &x0 - &x1;
        assert_eq!(*y.data(), (-1.0).into());
    }

    #[test]
    #[should_panic]
    fn error_sub_ref_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = &x0 - &x1;
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

    #[test]
    fn div_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = x0 / x1;
        assert_eq!(*y.data(), (2.0 / 3.0).into());
    }

    #[test]
    #[should_panic]
    fn error_div_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = x0 / x1;
    }

    #[test]
    fn div_ref_normal() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3.0);
        let y = &x0 / &x1;
        assert_eq!(*y.data(), (2.0 / 3.0).into());
    }

    #[test]
    #[should_panic]
    fn error_div_ref_different_type() {
        let x0 = Variable::from(2.0);
        let x1 = Variable::from(3);
        let _ = &x0 / &x1;
    }

    #[test]
    fn neg_normal() {
        let x = Variable::from(2.0);
        let y = -x;
        assert_eq!(*y.data(), (-2.0).into());
    }

    #[test]
    #[should_panic]
    fn error_neg_supported_type() {
        let x = Variable::from(true);
        let _ = -x;
    }

    #[test]
    fn neg_ref_normal() {
        let x = Variable::from(2.0);
        let y = -&x;
        assert_eq!(*y.data(), (-2.0).into());
    }

    #[test]
    #[should_panic]
    fn error_neg_ref_supported_type() {
        let x = Variable::from(true);
        let _ = -&x;
    }
}
