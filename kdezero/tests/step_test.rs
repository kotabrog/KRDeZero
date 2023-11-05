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

    println!("{}", dx2[0]);
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

    println!("{}", x.grad_result()?);
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

    println!("{}", x.grad_result()?);

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
fn step13() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::{add, square};

    let x = Variable::from(2.0);
    let y = Variable::from(3.0);
    let mut z = add(&square(&x)?, &square(&y)?)
        ?;
    z.backward()?;
    println!("{:?}", z.data());
    println!("{}", x.grad_result()?);
    println!("{}", y.grad_result()?);
    assert_eq!(*z.data(), 13.0.into());
    assert_eq!(*x.grad_result()?.data(), 4.0.into());
    assert_eq!(*y.grad_result()?.data(), 6.0.into());

    Ok(())
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

#[test]
fn step20() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::{add, mul};

    let a = Variable::from(3.0);
    let b = Variable::from(2.0);
    let c = Variable::from(1.0);

    let mut y = add(&mul(&a, &b)?, &c)?;
    y.backward()?;

    println!("{:?}", y.data());
    assert_eq!(*y.data(), 7.0.into());
    println!("{:?}", a.grad_result()?.data());
    assert_eq!(*a.grad_result()?.data(), 2.0.into());
    println!("{:?}", b.grad_result()?.data());
    assert_eq!(*b.grad_result()?.data(), 3.0.into());

    let a = Variable::from(3.0);
    let b = Variable::from(2.0);
    let c = Variable::from(1.0);

    let mut y = &a * &b + c;
    y.backward()?;

    println!("{:?}", y.data());
    assert_eq!(*y.data(), 7.0.into());
    println!("{:?}", a.grad_result()?.data());
    assert_eq!(*a.grad_result()?.data(), 2.0.into());
    println!("{:?}", b.grad_result()?.data());
    assert_eq!(*b.grad_result()?.data(), 3.0.into());

    Ok(())
}

#[test]
fn step21() {
    use ktensor::Tensor;
    use kdezero::Variable;

    let x = Variable::from(2.0);
    let y = x + Tensor::scalar(3.0).into();
    println!("{:?}", y.data());
    assert_eq!(*y.data(), 5.0.into());

    let x = Variable::from(2.0);
    let y = x + 3.0.into();
    println!("{:?}", y.data());
    assert_eq!(*y.data(), 5.0.into());
}

#[test]
fn step22() {
    use kdezero::Variable;
    use kdezero::function::pow;

    let x = Variable::from(2.0);
    let y = -x;
    println!("{}", y);
    assert_eq!(*y.data(), (-2.0).into());

    let x0 = Variable::from(2.0);
    let x1 = Variable::from(3.0);
    let y = x0 - x1;
    println!("{}", y);
    assert_eq!(*y.data(), (-1.0).into());

    let x0 = Variable::from(2.0);
    let x1 = Variable::from(3.0);
    let y = x0 / x1;
    println!("{}", y);
    assert_eq!(*y.data(), (2.0 / 3.0).into());

    let x = Variable::from(2.0);
    let y = pow(&x, 3.0).unwrap();
    println!("{}", y);
    assert_eq!(*y.data(), 8.0.into());
}

#[test]
fn step24() -> Result<()> {
    use ktensor::Tensor;
    use kdezero::Variable;
    use kdezero::function::pow;
    use kdezero::test_utility::assert_approx_eq_tensor;

    fn sphere(x: &Variable, y: &Variable) -> Result<Variable> {
        Ok(pow(x, 2.0)? + pow(y, 2.0)?)
    }

    let x = Variable::from(1.0);
    let y = Variable::from(1.0);
    let mut z = sphere(&x, &y)?;
    z.backward()?;
    println!("{:?}", x.grad_result()?.data());
    assert_eq!(*x.grad_result()?.data(), 2.0.into());
    println!("{:?}", y.grad_result()?.data());
    assert_eq!(*y.grad_result()?.data(), 2.0.into());

    fn matyas(x: &Variable, y: &Variable) -> Result<Variable> {
        let z0 = (pow(x, 2.0)? + pow(y, 2.0)?) * 0.26.into();
        let z1 = x * y * 0.48.into();
        Ok(z0 - z1)
    }

    let x = Variable::from(1.0);
    let y = Variable::from(1.0);
    let mut z = matyas(&x, &y)?;
    z.backward()?;
    println!("{:?}", x.grad_result()?.data());
    assert_approx_eq_tensor(
        x.grad_result()?.data().to_f64_tensor()?,
        &Tensor::scalar(0.040000000000000036), 1e-4);
    println!("{:?}", y.grad_result()?.data());
    assert_approx_eq_tensor(
        y.grad_result()?.data().to_f64_tensor()?,
        &Tensor::scalar(0.040000000000000036), 1e-4);

    Ok(())
}

#[test]
fn step26() -> Result<()> {
    use std::fs::create_dir;
    use kdezero::Variable;
    use kdezero::function::pow;
    use kdezero::plot_dot_graph;

    fn matyas(x: &Variable, y: &Variable) -> Result<Variable> {
        let z0 = (pow(x, 2.0)? + pow(y, 2.0)?) * 0.26.into();
        let z1 = x * y * 0.48.into();
        Ok(z0 - z1)
    }

    let mut x = Variable::from(1.0);
    let mut y = Variable::from(1.0);
    let mut z = matyas(&x, &y)?;
    z.backward()?;

    x.set_name("x");
    y.set_name("y");
    z.set_name("z");

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    plot_dot_graph(&z, "output/step26", true, false)?;
    plot_dot_graph(&z, "output/step26_verbose", true, true)?;

    Ok(())
}

#[test]
fn step27() -> Result<()> {
    use std::f64::consts::FRAC_PI_4;
    use std::fs::create_dir;
    use ktensor::Tensor;
    use kdezero::Variable;
    use kdezero::function::{sin, pow};
    use kdezero::plot_dot_graph;
    use kdezero::test_utility::assert_approx_eq_tensor;

    let x = Variable::from(FRAC_PI_4);
    let mut y = sin(&x)?;
    y.backward()?;

    println!("{}", y);
    assert_eq!(*y.data(), FRAC_PI_4.sin().into());

    println!("{}", x.grad_result()?);
    assert_eq!(*x.grad_result()?.data(), FRAC_PI_4.cos().into());

    fn my_sin(x: &Variable, threshold: f64) -> Result<Variable> {
        let mut y = Variable::from(0.0);
        for i in 0..100000 {
            let mut c = if i % 2 == 0 { 1.0 } else { -1.0 };
            for j in 1..=(2 * i + 1) {
                c /= j as f64;
            }
            let t = pow(x, 2.0 * i as f64 + 1.0)? * c.into();
            y = &y + &t;
            if t.data().to_f64_tensor()?.to_scalar()?.abs() < threshold {
                break;
            }
        }
        Ok(y.into())
    }

    let x = Variable::new_with_name(FRAC_PI_4.into(), "x");
    let mut y = my_sin(&x, 1e-4)?;
    y.set_name("y");
    y.backward()?;

    println!("{}", y);
    assert_approx_eq_tensor(
        y.data().to_f64_tensor()?,
        &Tensor::scalar(FRAC_PI_4.sin()), 1e-4);
    println!("{}", x.grad_result()?);
    assert_approx_eq_tensor(
        x.grad_result()?.data().to_f64_tensor()?,
        &Tensor::scalar(FRAC_PI_4.cos()), 1e-4);

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    plot_dot_graph(&y, "output/step27", true, false)?;

    Ok(())
}

#[test]
fn step28() -> Result<()> {
    use kdezero::{Variable, VariableData};
    use kdezero::function::pow;

    fn resenbrock(x0: &Variable, x1: &Variable) -> Result<Variable> {
        let y = pow(&(x1 - &pow(x0, 2.0)?), 2.0)? * 100.0.into()
            + pow(&(x0 - &1.0.into()), 2.0)?;
        Ok(y)
    }

    let mut x0 = Variable::from(0.0);
    let mut x1 = Variable::from(2.0);
    let lr = 0.001;
    let lr = VariableData::from(lr);
    let iters = 1000;

    for i in 0..iters {
        let mut y = resenbrock(&x0, &x1)?;
        x0.clear_grad();
        x1.clear_grad();
        y.backward()?;

        let new_x0 =
            x0.data()
            .sub(&x0.grad_result()?.data().mul(&lr)?)?;
        x0.set_data(new_x0);

        let new_x1 =
            x1.data()
            .sub(&x1.grad_result()?.data().mul(&lr)?)?;
        x1.set_data(new_x1);

        if i % 100 == 0 {
            println!("{} {:.10} {:.10}", i, x0.data(), x1.data());
        }
    }

    Ok(())
}

#[test]
fn step33() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::pow;

    fn f(x: &Variable) -> Result<Variable> {
        let y = pow(x, 4.0)? - pow(x, 2.0)? * 2.0.into();
        Ok(y)
    }

    let mut x = Variable::from(2.0);
    let mut y = f(&x)?;
    y.backward_create_graph()?;

    println!("{}", x.grad_result()?);
    assert_eq!(*x.grad_result()?.data(), 24.0.into());

    let mut gx = x.grad_result()?;
    x.clear_grad();
    gx.backward()?;

    println!("{}", x.grad_result()?);
    assert_eq!(*x.grad_result()?.data(), 44.0.into());

    let mut x = Variable::from(2.0);
    let iters = 10;

    for i in 0..iters {
        let mut y = f(&x)?;
        x.clear_grad();
        y.backward_create_graph()?;

        let mut gx = x.grad_result()?;
        x.clear_grad();
        gx.backward()?;
        let gx2 = x.grad_result()?;

        let new_x = x.data()
            .sub(&gx.data().div(&gx2.data())?)?;
        x.set_data(new_x);

        println!("{} {:.10}", i, x.data());
    }

    Ok(())
}

#[test]
fn step34() -> Result<()> {
    use std::fs::create_dir;
    use plotters::prelude::*;
    use ktensor::Tensor;
    use kdezero::Variable;
    use kdezero::function::sin;

    let mut x = Variable::from(1.0);
    let mut y = sin(&x)?;
    y.backward_create_graph()?;

    for _ in 0..3 {
        let mut gx = x.grad_result()?;
        x.clear_grad();
        gx.backward_create_graph()?;
        println!("{}", x.grad_result()?);
    }

    let mut x = Variable::from(Tensor::linspace(-7.0, 7.0, 200));
    let mut y = sin(&x)?;
    y.backward_create_graph()?;

    let mut logs = Vec::new();
    logs.push(y.data().to_f64_tensor()?.to_vector()?);

    for _ in 0..3 {
        let mut gx = x.grad_result()?;
        x.clear_grad();
        gx.backward_create_graph()?;
        println!("{}", x.grad_result()?);
        logs.push(gx.data().to_f64_tensor()?.to_vector()?);
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let root = BitMapBackend::new("output/step34.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            -7.0f64..7.0f64,
            -1.5f64..1.5f64
        )?;

    chart.configure_mesh().draw()?;

    let styles = [
        ShapeStyle::from(&RED).filled(),
        ShapeStyle::from(&GREEN).filled(),
        ShapeStyle::from(&BLUE).filled(),
        ShapeStyle::from(&CYAN).filled(),
    ];

    let xs = Tensor::linspace(-7.0, 7.0, 200)
        .to_vector()?;

    let labels = [
        "y=sin(x)",
        "y'",
        "y''",
        "y'''",
    ];

    for i in 0..4 {
        chart.draw_series(
            LineSeries::new(
                logs[i].iter().enumerate().map(|(j, y)| (xs[j], *y)),
                styles[i].clone()
            )
        )?.label(labels[i])
        .legend(move |(x, y)|
            Rectangle::new([(x, y - 5), (x + 10, y + 5)], styles[i].clone()));
    }

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
