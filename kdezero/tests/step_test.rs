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

#[test]
fn step9() -> Result<()> {
    use kdezero::{Variable, FunctionContent};
    use kdezero::function::{square, exp};
    use kdezero::test_utility::{numerical_diff, assert_approx_eq_tensor};

    #[derive(Debug)]
    struct SES {}

    impl FunctionContent for SES {
        fn forward(&self, xs: Vec<&Variable>) -> Result<Vec<Variable>> {
            let x = xs[0];
            let y = square(&exp(&square(x)?)?)?;
            Ok(vec![y])
        }
    }

    let x = Variable::from(0.5);
    let mut f = SES {};
    let dx = numerical_diff(
        &mut f, &x, 1e-4)?;

    let x = Variable::from(0.5);
    let mut y = square(&exp(&square(&x)?)?)?;

    y.backward()?;

    println!("{:?}", x.grad_result()?);

    assert_approx_eq_tensor(
        dx.data().to_f64_tensor()?,
        x.grad_result()?.data().to_f64_tensor()?, 1e-4);

    Ok(())
}

#[test]
fn step11() {
    use kdezero::Variable;
    use kdezero::function::add;

    let x0 = Variable::from(2);
    let x1 = Variable::from(3);
    let y = add(&x0, &x1).unwrap();
    println!("{:?}", y.data());
    assert_eq!(*y.data(), 5.into());
}

#[test]
fn step13() {
    use kdezero::Variable;
    use kdezero::function::{add, square};

    let x = Variable::from(2.0);
    let y = Variable::from(3.0);
    let mut z = add(&square(&x).unwrap(), &square(&y).unwrap())
        .unwrap();
    z.backward().unwrap();
    println!("{:?}", z.data());
    println!("{:?}", x.grad_result().unwrap());
    println!("{:?}", y.grad_result().unwrap());
    assert_eq!(*z.data(), 13.0.into());
    assert_eq!(*x.grad_result().unwrap().data(), 4.0.into());
    assert_eq!(*y.grad_result().unwrap().data(), 6.0.into());
}

#[test]
fn step14() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::add;

    let x = Variable::from(3.0);
    let mut y = add(&x, &x)?;
    y.backward()?;
    println!("{:?}", x.grad_result()?.data());
    assert_eq!(*x.grad_result()?.data(), 2.0.into());

    let x = Variable::from(3.0);
    let mut y = add(&add(&x, &x)?, &x)?;
    y.backward()?;
    println!("{:?}", x.grad_result()?.data());
    assert_eq!(*x.grad_result()?.data(), 3.0.into());

    let mut x = Variable::from(3.0);
    let mut y = add(&x, &x)?;
    y.backward()?;
    println!("{:?}", x.grad_result()?.data());

    x.clear_grad();
    let mut y = add(&add(&x, &x)?, &x)?;
    y.backward()?;
    println!("{:?}", x.grad_result()?.data());
    assert_eq!(*x.grad_result()?.data(), 3.0.into());

    Ok(())
}

#[test]
fn step16() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::{add, square};

    let x = Variable::from(2.0);
    let a = square(&x)?;
    let mut y = add(&square(&a)?, &square(&a)?)?;
    y.backward()?;

    println!("{:?}", y.data());
    println!("{:?}", x.grad_result()?.data());
    assert_eq!(*y.data(), 32.0.into());
    assert_eq!(*x.grad_result()?.data(), 64.0.into());

    Ok(())
}

#[test]
fn step18() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::{add, square};
    use kdezero::no_grad;

    let x0 = Variable::from(1.0);
    let x1 = Variable::from(1.0);
    let t = add(&x0, &x1)?;
    let mut y = add(&x0, &t)?;
    y.backward()?;

    println!("{:?}", y.grad_clone());
    assert!(y.is_grad_none());
    println!("{:?}", t.grad_clone());
    assert!(t.is_grad_none());
    println!("{:?}", x0.grad_result()?.data());
    assert_eq!(*x0.grad_result()?.data(), 2.0.into());
    println!("{:?}", x1.grad_result()?.data());
    assert_eq!(*x1.grad_result()?.data(), 1.0.into());

    {
        let _guard = no_grad();
        let x = Variable::from(2.0);
        let mut y = square(&x)?;

        println!("{:?}", y.data());
        assert_eq!(*y.data(), 4.0.into());

        match y.backward() {
            Ok(_) => panic!("should not be here"),
            Err(e) => println!("{:?}", e),
        }
    }

    Ok(())
}

#[test]
fn step19() -> Result<()> {
    use ktensor::Tensor;
    use kdezero::Variable;

    let x = Variable::new_with_name(2.0.into(), "x");
    println!("{:?}", x.name());
    assert_eq!(*x.name(), "x");

    let x = Variable::from(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?
    );
    println!("{:?}", x.shape());
    assert_eq!(*x.shape(), vec![2, 3]);

    println!("{:?}", x.ndim());
    assert_eq!(x.ndim(), 2);

    println!("{:?}", x.size());
    assert_eq!(x.size(), 6);

    println!("{:?}", x.data_type());
    assert_eq!(*x.data_type(), *"f64");

    println!("{}", x);
    assert_eq!(format!("{}", x), "Variable(tensor([[1, 2, 3],\n                 [4, 5, 6]], type=f64))");

    Ok(())
}
