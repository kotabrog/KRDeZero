use anyhow::Result;
use crate::Variable;
use super::super::Function;
use super::super::function_helper::check_variable_count;

pub struct Exp {}

impl Exp {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function for Exp {
    fn forward(&self, xs: &[Variable]) -> Result<Vec<Variable>> {
        check_variable_count(xs, 1)?;
        let x = xs[0].data();
        let y = x.exp()?;
        Ok(vec![y.into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::KDeZeroError;

    #[test]
    fn exp_forward() -> Result<()> {
        let x = Variable::from(2.0);
        let y = Exp::new().forward(&vec![x])?;
        assert_eq!(y[0].data(), &2.0f64.exp().into());
        Ok(())
    }

    #[test]
    fn error_exp_forward_invalid_variable_count() -> Result<()> {
        let x = Variable::from(2.0);
        match Exp::new().forward(&vec![x.clone(), x]) {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>()?;
                assert_eq!(e, KDeZeroError::InvalidVariableCount(
                    1,
                    2,
                ));
            }
        }
        Ok(())
    }
}
