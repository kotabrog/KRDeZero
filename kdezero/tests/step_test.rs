use anyhow::Result;

#[test]
fn step1() {
    use kdezero::Variable;

    let data = 1.0;
    let mut x = Variable::from(data);
    println!("{:?}", x.data());

    assert_eq!(*x.data(), 1.0.into());

    x.set_data(2.0.into());
    println!("{:?}", x.data());

    assert_eq!(*x.data(), 2.0.into());
}

#[test]
fn step2() -> Result<()> {
    use kdezero::{Variable, FunctionContent};
    use kdezero::function::Square;

    let x = Variable::from(10);
    let f = Square::new();
    let y = f.forward(vec![&x])?;

    println!("{:?}", y[0].data());

    assert_eq!(*y[0].data(), 100.into());
    Ok(())
}

#[test]
fn step3() -> Result<()> {
    use kdezero::{Variable, FunctionContent};
    use kdezero::function::{Square, Exp};

    let x = Variable::from(0.5);
    let a = Square::new();
    let b = Exp::new();
    let c = Square::new();

    let y1 = a.forward(vec![&x])?;
    let y2 = b.forward(y1.iter().collect())?;
    let y3 = c.forward(y2.iter().collect())?;

    println!("{:?}", y3[0].data());
    assert_eq!(*y3[0].data(), 0.5f64.powi(2).exp().powi(2).into());

    Ok(())
}

#[test]
fn step4() -> Result<()> {
    use ktensor::Tensor;
    use kdezero::{Variable, FunctionContent};
    use kdezero::function::{Square, Exp};
    use kdezero::test_utility::{numerical_diff, assert_approx_eq_tensor};

    let x = Variable::from(2.0);
    let mut f = Square::new();
    let dy = numerical_diff(
        &mut f, &x, 1e-4)?;

    println!("{:?}", dy);
    assert_approx_eq_tensor(
        dy.data().to_f64_tensor()?, &Tensor::scalar(4.0), 1e-4);

    #[derive(Debug)]
    struct SES {}

    impl FunctionContent for SES {
        fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
            let y = Square::new().forward(xs)?;
            let y = Exp::new().forward(y.iter().collect())?;
            Square::new().forward(y.iter().collect())
        }
    }

    let x = Variable::from(0.5);
    let mut f = SES {};
    let dy = numerical_diff(
        &mut f, &x, 1e-4)?;

    println!("{:?}", dy);
    assert_approx_eq_tensor(
        dy.data().to_f64_tensor()?,
        &Tensor::scalar(3.2974426293330694), 1e-4);

    Ok(())
}

#[test]
fn step6() -> Result<()> {
    use kdezero::{Variable, FunctionContent, Function};
    use kdezero::function::{Square, Exp};
    use kdezero::test_utility::{numerical_diff, assert_approx_eq_tensor};

    #[derive(Debug)]
    struct SES {}

    impl FunctionContent for SES {
        fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
            let y = Square::new().forward(xs)?;
            let y = Exp::new().forward(y.iter().collect())?;
            Square::new().forward(y.iter().collect())
        }
    }

    let x = Variable::from(0.5);
    let mut f = SES {};
    let dx1 = numerical_diff(
        &mut f, &x, 1e-4)?;

    let mut a = Function::new(Square::new());
    let mut b = Function::new(Exp::new());
    let mut c = Function::new(Square::new());

    let y1 = a.forward(&[x])?;
    let y2 = b.forward(&y1)?;
    let _y3 = c.forward(&y2)?;

    let dx = Variable::from(1.0);
    let dx = c.backward(&[dx])?;
    let dx = b.backward(&dx)?;
    let dx2 = a.backward(&dx)?;

    println!("{:?}", dx2);
    assert_approx_eq_tensor(
        dx1.data().to_f64_tensor()?,
        dx2[0].data().to_f64_tensor()?, 1e-4);
    Ok(())
}

#[test]
fn step8() -> Result<()> {
    use kdezero::{Variable, FunctionContent, Function};
    use kdezero::function::{Square, Exp};
    use kdezero::test_utility::{numerical_diff, assert_approx_eq_tensor};

    #[derive(Debug)]
    struct SES {}

    impl FunctionContent for SES {
        fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
            let y = Square::new().forward(xs)?;
            let y = Exp::new().forward(y.iter().collect())?;
            Square::new().forward(y.iter().collect())
        }
    }

    let x = Variable::from(0.5);
    let mut f = SES {};
    let dx = numerical_diff(
        &mut f, &x, 1e-4)?;

    let mut a = Function::new(Square::new());
    let mut b = Function::new(Exp::new());
    let mut c = Function::new(Square::new());

    let y = a.forward(&[x.clone()])?;
    let y = b.forward(&y)?;
    let mut y = c.forward(&y)?;

    y[0].set_grad(Variable::from(1.0));

    y[0].backward()?;

    println!("{:?}", x.grad_result()?);
    assert_approx_eq_tensor(
        dx.data().to_f64_tensor()?,
        x.grad_result()?.data().to_f64_tensor()?, 1e-4);

    Ok(())
}
