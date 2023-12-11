use anyhow::Result;
use plotters::prelude::*;

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

#[test]
fn step35() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::tanh;
    use kdezero::plot_dot_graph;

    let mut x = Variable::from(1.0);
    let mut y = tanh(&x)?;
    x.set_name("x");
    y.set_name("y");
    y.backward_create_graph()?;

    let iters = 2;

    for _ in 0..iters {
        let mut gx = x.grad_result()?;
        x.clear_grad();
        gx.backward_create_graph()?;
    }

    let mut gx = x.grad_result()?;
    gx.set_name("gx");
    plot_dot_graph(&gx, "output/step35", true, false)?;

    Ok(())
}

#[test]
fn step36() -> Result<()> {
    use kdezero::Variable;
    use kdezero::function::pow;

    let mut x = Variable::from(2.0);
    let mut y = pow(&x, 2.0)?;
    y.backward_create_graph()?;
    let gx = x.grad_result()?;
    x.clear_grad();

    let mut z = pow(&gx, 3.0)? + y;
    z.backward()?;
    println!("{}", x.grad_result()?);
    assert_eq!(*x.grad_result()?.data(), 100.0.into());

    Ok(())
}

#[test]
fn step38() -> Result<()> {
    use ktensor::Tensor;
    use kdezero::Variable;
    use kdezero::function::{reshape, transpose};

    let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
    let mut y = reshape(&x, &[6])?;
    y.backward()?;
    println!("{}", x.grad_result()?);
    assert_eq!(
        *x.grad_result()?.data(),
        Tensor::<f64>::ones(vec![2, 3]).into());

    let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
    let mut y = transpose(&x)?;
    y.backward()?;
    println!("{}", x.grad_result()?);
    assert_eq!(
        *x.grad_result()?.data(),
        Tensor::<f64>::ones(vec![2, 3]).into());

    Ok(())
}

#[test]
fn step40() -> Result<()> {
    use ktensor::Tensor;
    use kdezero::Variable;
    use kdezero::function::{sum_keepdims, sum_axis};

    let x = Variable::new(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3])?.into());
    let mut y = sum_axis(&x, vec![0], false)?;
    y.backward()?;
    println!("{}", y);
    assert_eq!(
        *y.data(),
        Tensor::new(vec![5.0, 7.0, 9.0], vec![3])?.into());
    println!("{}", x.grad_result()?);
    assert_eq!(
        *x.grad_result()?.data(),
        Tensor::<f64>::ones(vec![2, 3]).into());


    let x = Variable::new(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3])?.into());
    let mut y = sum_keepdims(&x)?;
    y.backward()?;
    println!("{}", y);
    assert_eq!(
        *y.data(),
        Tensor::new(vec![21.0], vec![1, 1])?.into());
    println!("{}", x.grad_result()?);
    assert_eq!(
        *x.grad_result()?.data(),
        Tensor::<f64>::ones(vec![2, 3]).into());

    Ok(())
}

#[test]
fn step41() -> Result<()> {
    use ktensor::Tensor;
    use kdezero::Variable;
    use kdezero::function::matmul;

    let x = Variable::new(Tensor::<f64>::arrange([2, 3])?.into());
    let w = Variable::new(Tensor::<f64>::arrange([3, 4])?.into());
    let mut y = matmul(&x, &w)?;
    y.backward()?;

    println!("{:?}", x.grad_result()?.shape());
    assert_eq!(*x.grad_result()?.shape(), [2, 3]);
    println!("{:?}", w.grad_result()?.shape());
    assert_eq!(*w.grad_result()?.shape(), [3, 4]);
    Ok(())
}

fn plot_data_and_line(data: &[(f64, f64)], file_name: &str, line_points: Vec<(f64, f64)>) -> Result<()> {
    let root = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data.iter().map(|&(_, y)| y).fold(f64::INFINITY, f64::min);

    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin)
    )?;

    chart.configure_mesh().draw()?;

    let shape_style = ShapeStyle::from(&BLUE).filled();
    chart.draw_series(data.iter().map(|&(x, y)| Circle::new((x, y), 5, shape_style)))?;

    chart.draw_series(LineSeries::new(line_points, &RED))?;

    root.present()?;

    Ok(())
}

#[test]
fn step42() -> Result<()> {
    use std::fs::create_dir;
    use ktensor::{Tensor, tensor::TensorRng};
    use kdezero::Variable;
    use kdezero::function::{matmul, broadcast_to, mean_squared_error};

    fn predict(x: &Variable, w: &Variable, b: &Variable) -> Result<Variable> {
        let y = matmul(&x, &w)? + broadcast_to(&b, &[x.shape()[0], 1])?;
        Ok(y)
    }

    // fn mean_squared_error(
    //     y_pred: &Variable, y_true: &Variable) -> Result<Variable> {
    //     let n = y_pred.shape()[0] as f64;
    //     let diff = y_pred - y_true;
    //     let loss = sum_all(&pow(&diff, 2.0)?)? / n.into();
    //     Ok(loss)
    // }

    let mut rng = TensorRng::new();
    let x_data = rng.gen::<f64, _>(vec![100, 1]);
    let y_data = &x_data * 2.0 + 5.0 + rng.gen::<f64, _>(vec![100, 1]);

    let x = Variable::new(x_data.clone().into());
    let y = Variable::new(y_data.clone().into());

    let mut w = Variable::new(Tensor::<f64>::zeros(vec![1, 1]).into());
    let mut b = Variable::new(Tensor::<f64>::zeros(vec![1, 1]).into());

    let lr = 0.1;
    let iters = 100;

    for i in 0..iters {
        let y_pred = predict(&x, &w, &b)?;
        let mut loss = mean_squared_error(&y_pred, &y)?;
        w.clear_grad();
        b.clear_grad();
        loss.backward()?;

        let new_w = w.data()
            .sub(&w.grad_result()?.data().scalar_mul(lr)?)?;
        w.set_data(new_w);

        let new_b = b.data()
            .sub(&b.grad_result()?.data().scalar_mul(lr)?)?;
        b.set_data(new_b);

        if i % 10 == 0 || i == iters - 1 {
            println!("{} {:.10}", i, loss);
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let data: Vec<(f64, f64)> = x_data.iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);

    let w = w.data().to_f64_tensor()?.to_scalar()?;
    let b = b.data().to_f64_tensor()?.to_scalar()?;

    let line_points: Vec<(f64, f64)> = (0..=100)
        .map(|x| x as f64 / 100.0 * (x_max - x_min) + x_min)
        .map(|x| (x, w * x + b))
        .collect();

    plot_data_and_line(
        &data,
        "output/step42.png",
        line_points
    )?;

    Ok(())
}

#[test]
fn step43() -> Result<()> {
    use std::fs::create_dir;
    use ktensor::{Tensor, tensor::TensorRng};
    use kdezero::Variable;
    use kdezero::function::{linear, sigmoid, broadcast_to, mean_squared_error};

    fn predict(x: &Variable, w0: &Variable, w1: &Variable, b0: &Variable, b1: &Variable) -> Result<Variable> {
        let b0 = broadcast_to(b0, &[x.shape()[0], w0.shape()[1]])?;
        let y = linear(x, w0, Some(&b0))?;
        let y = sigmoid(&y)?;
        let b1 = broadcast_to(b1, &[x.shape()[0], w1.shape()[1]])?;
        let y = linear(&y, w1, Some(&b1))?;
        Ok(y)
    }

    let mut rng = TensorRng::new();
    let x_data = rng.gen::<f64, _>(vec![100, 1]);
    let y_data = (&x_data * 2.0 * std::f64::consts::PI).sin() + rng.gen::<f64, _>(vec![100, 1]);

    let x = Variable::new(x_data.clone().into());
    let y = Variable::new(y_data.clone().into());

    let mut w0 = Variable::new(rng.gen::<f64, _>(vec![1, 10]).into());
    let mut b0 = Variable::new(Tensor::<f64>::zeros(vec![10]).into());
    let mut w1 = Variable::new(rng.gen::<f64, _>(vec![10, 1]).into());
    let mut b1 = Variable::new(Tensor::<f64>::zeros(vec![]).into());

    let lr = 0.2;
    let iters = 100;
    // let iters = 10000;

    for i in 0..iters {
        let y_pred = predict(&x, &w0, &w1, &b0, &b1)?;
        let mut loss = mean_squared_error(&y_pred, &y)?;
        w0.clear_grad();
        b0.clear_grad();
        w1.clear_grad();
        b1.clear_grad();
        loss.backward()?;

        let new_w = w0.data()
            .sub(&w0.grad_result()?.data().scalar_mul(lr)?)?;
        w0.set_data(new_w);

        let new_w = w1.data()
            .sub(&w1.grad_result()?.data().scalar_mul(lr)?)?;
        w1.set_data(new_w);

        let new_b = b0.data()
            .sub(&b0.grad_result()?.data().scalar_mul(lr)?)?;
        b0.set_data(new_b);

        let new_b = b1.data()
            .sub(&b1.grad_result()?.data().scalar_mul(lr)?)?;
        b1.set_data(new_b);

        if i % (iters / 10) == 0 || i == iters - 1 {
            println!("{} {:.10}", i, loss);
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let data: Vec<(f64, f64)> = x_data.iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);

    let line_x: Vec<f64> = (0..=100)
        .map(|x| x as f64 / 100.0 * (x_max - x_min) + x_min)
        .collect();
    let y = predict(
        &Variable::new(Tensor::new(line_x.clone(), vec![line_x.len(), 1])?.into()),
        &w0, &w1, &b0, &b1
    )?;
    let line_y = y.data().to_f64_tensor()?.get_data().clone();
    let line_points = line_x.iter()
        .zip(line_y.iter())
        .map(|(&x, &y)| (x, y)).collect();

    plot_data_and_line(
        &data,
        "output/step43.png",
        line_points
    )?;

    Ok(())
}

#[test]
fn step44() -> Result<()> {
    use std::fs::create_dir;
    use ktensor::{Tensor, tensor::TensorRng};
    use kdezero::{Variable, VariableType};
    use kdezero::function::{sigmoid, mean_squared_error};
    use kdezero::{Layer, layer::Linear};

    fn predict(x: &Variable, l1: &Layer, l2: &Layer) -> Result<Variable> {
        let y = l1.forward(&[x.clone()])?.remove(0);
        let y = sigmoid(&y)?;
        let y = l2.forward(&[y])?.remove(0);
        Ok(y)
    }

    let mut l1 = Layer::new(Linear::new(
        1, 10, true, VariableType::F64)?);
    let mut l2 = Layer::new(Linear::new(
        10, 1, true, VariableType::F64)?);

    let mut rng = TensorRng::new();
    let x_data = rng.gen::<f64, _>(vec![100, 1]);
    let y_data = (&x_data * 2.0 * std::f64::consts::PI).sin() + rng.gen::<f64, _>(vec![100, 1]);

    let x = Variable::new(x_data.clone().into());
    let y = Variable::new(y_data.clone().into());

    let lr = 0.2;
    let iters = 100;
    // let iters = 10000;

    for i in 0..iters {
        let y_pred = predict(&x, &l1, &l2)?;
        let mut loss = mean_squared_error(&y_pred, &y)?;
        l1.clear_grads();
        l2.clear_grads();
        loss.backward()?;

        let l1_params = l1.get_params();
        let mut w0 = l1_params["weight"].clone();
        let mut b0 = l1_params["bias"].clone();
        let l2_params = l2.get_params();
        let mut w1 = l2_params["weight"].clone();
        let mut b1 = l2_params["bias"].clone();

        let new_w = w0.data()
            .sub(&w0.grad_result()?.data().scalar_mul(lr)?)?;
        w0.set_data(new_w);

        let new_w = w1.data()
            .sub(&w1.grad_result()?.data().scalar_mul(lr)?)?;
        w1.set_data(new_w);

        let new_b = b0.data()
            .sub(&b0.grad_result()?.data().scalar_mul(lr)?)?;
        b0.set_data(new_b);

        let new_b = b1.data()
            .sub(&b1.grad_result()?.data().scalar_mul(lr)?)?;
        b1.set_data(new_b);

        if i % (iters / 10) == 0 || i == iters - 1 {
            println!("{} {:.10}", i, loss);
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let data: Vec<(f64, f64)> = x_data.iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);

    let line_x: Vec<f64> = (0..=100)
        .map(|x| x as f64 / 100.0 * (x_max - x_min) + x_min)
        .collect();

    let y = predict(
        &Variable::new(Tensor::new(line_x.clone(), vec![line_x.len(), 1])?.into()),
        &l1, &l2
    )?;
    let line_y = y.data().to_f64_tensor()?.get_data().clone();
    let line_points = line_x.iter()
        .zip(line_y.iter())
        .map(|(&x, &y)| (x, y)).collect();

    plot_data_and_line(
        &data,
        "output/step44.png",
        line_points
    )?;

    Ok(())
}

#[test]
fn step45_1() -> Result<()> {
    use std::fs::create_dir;
    use ktensor::{Tensor, tensor::TensorRng};
    use kdezero::{Variable, Model};
    use kdezero::function::mean_squared_error;
    use kdezero::model::TwoLayerNet;

    let layer = TwoLayerNet::new(1, 10, 1)?;
    let mut model = Model::new(layer);

    let mut rng = TensorRng::new();
    let x_data = rng.gen::<f64, _>(vec![100, 1]);
    let y_data = (&x_data * 2.0 * std::f64::consts::PI).sin() + rng.gen::<f64, _>(vec![100, 1]);

    let x = Variable::new(x_data.clone().into());
    let y = Variable::new(y_data.clone().into());

    let lr = 0.2;
    let iters = 100;
    // let iters = 10000;

    for i in 0..iters {
        let y_pred = model.forward(&[x.clone()])?.remove(0);
        let mut loss = mean_squared_error(&y_pred, &y)?;
        model.clear_grads();
        loss.backward()?;

        let params = model.get_params();

        let mut w0 = params["l1.weight"].clone();
        let mut b0 = params["l1.bias"].clone();
        let mut w1 = params["l2.weight"].clone();
        let mut b1 = params["l2.bias"].clone();

        let new_w = w0.data()
            .sub(&w0.grad_result()?.data().scalar_mul(lr)?)?;
        w0.set_data(new_w);

        let new_w = w1.data()
            .sub(&w1.grad_result()?.data().scalar_mul(lr)?)?;
        w1.set_data(new_w);

        let new_b = b0.data()
            .sub(&b0.grad_result()?.data().scalar_mul(lr)?)?;
        b0.set_data(new_b);

        let new_b = b1.data()
            .sub(&b1.grad_result()?.data().scalar_mul(lr)?)?;
        b1.set_data(new_b);

        if i % (iters / 10) == 0 || i == iters - 1 {
            println!("{} {:.10}", i, loss);
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let data: Vec<(f64, f64)> = x_data.iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);

    let line_x: Vec<f64> = (0..=100)
        .map(|x| x as f64 / 100.0 * (x_max - x_min) + x_min)
        .collect();

    let y = model.forward(
        &[Variable::new(
                Tensor::new(
                        line_x.clone(),
                        vec![line_x.len(), 1])?.into())]
    )?.remove(0);
    let line_y = y.data().to_f64_tensor()?.get_data().clone();
    let line_points = line_x.iter()
        .zip(line_y.iter())
        .map(|(&x, &y)| (x, y)).collect();

    plot_data_and_line(
        &data,
        "output/step45_1.png",
        line_points
    )?;

    model.plot(&[x], "output/step45_1_plot")?;

    Ok(())
}

#[test]
fn step45_2() -> Result<()> {
    use std::fs::create_dir;
    use ktensor::{Tensor, tensor::TensorRng};
    use kdezero::{Variable, Model};
    use kdezero::function::{mean_squared_error, sigmoid};
    use kdezero::model::MLP;

    let layer = MLP::new(&[1, 10, 1], sigmoid)?;
    let mut model = Model::new(layer);

    let mut rng = TensorRng::new();
    let x_data = rng.gen::<f64, _>(vec![100, 1]);
    let y_data = (&x_data * 2.0 * std::f64::consts::PI).sin() + rng.gen::<f64, _>(vec![100, 1]);

    let x = Variable::new(x_data.clone().into());
    let y = Variable::new(y_data.clone().into());

    let lr = 0.2;
    let iters = 100;
    // let iters = 10000;

    for i in 0..iters {
        let y_pred = model.forward(&[x.clone()])?.remove(0);
        let mut loss = mean_squared_error(&y_pred, &y)?;
        model.clear_grads();
        loss.backward()?;

        let params = model.get_params();

        let mut w0 = params["l1.weight"].clone();
        let mut b0 = params["l1.bias"].clone();
        let mut w1 = params["l2.weight"].clone();
        let mut b1 = params["l2.bias"].clone();

        let new_w = w0.data()
            .sub(&w0.grad_result()?.data().scalar_mul(lr)?)?;
        w0.set_data(new_w);

        let new_w = w1.data()
            .sub(&w1.grad_result()?.data().scalar_mul(lr)?)?;
        w1.set_data(new_w);

        let new_b = b0.data()
            .sub(&b0.grad_result()?.data().scalar_mul(lr)?)?;
        b0.set_data(new_b);

        let new_b = b1.data()
            .sub(&b1.grad_result()?.data().scalar_mul(lr)?)?;
        b1.set_data(new_b);

        if i % (iters / 10) == 0 || i == iters - 1 {
            println!("{} {:.10}", i, loss);
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let data: Vec<(f64, f64)> = x_data.iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);

    let line_x: Vec<f64> = (0..=100)
        .map(|x| x as f64 / 100.0 * (x_max - x_min) + x_min)
        .collect();

    let y = model.forward(
        &[Variable::new(
                Tensor::new(
                        line_x.clone(),
                        vec![line_x.len(), 1])?.into())]
    )?.remove(0);
    let line_y = y.data().to_f64_tensor()?.get_data().clone();
    let line_points = line_x.iter()
        .zip(line_y.iter())
        .map(|(&x, &y)| (x, y)).collect();

    plot_data_and_line(
        &data,
        "output/step45_2.png",
        line_points
    )?;

    model.plot(&[x], "output/step45_2_plot")?;

    Ok(())
}

#[test]
fn step46() -> Result<()> {
    use std::fs::create_dir;
    use ktensor::{Tensor, tensor::TensorRng};
    use kdezero::{Variable, Model, Optimizer};
    use kdezero::function::{mean_squared_error, sigmoid};
    use kdezero::model::MLP;
    // use kdezero::optimizer::SGD;
    use kdezero::optimizer::MomentumSGD;

    let layer = MLP::new(&[1, 10, 1], sigmoid)?;
    let model = Model::new(layer);
    // let opt_content = SGD::new(0.2);
    let opt_content = MomentumSGD::new(0.01, 0.9);
    let mut optimizer = Optimizer::new(opt_content);
    optimizer.set_model(model);

    let mut rng = TensorRng::new();
    let x_data = rng.gen::<f64, _>(vec![100, 1]);
    let y_data = (&x_data * 2.0 * std::f64::consts::PI).sin() + rng.gen::<f64, _>(vec![100, 1]);

    let x = Variable::new(x_data.clone().into());
    let y = Variable::new(y_data.clone().into());

    let iters = 100;
    // let iters = 10000;

    for i in 0..iters {
        let model = optimizer.get_model_mut_result()?;
        let y_pred = model.forward(&[x.clone()])?.remove(0);
        let mut loss = mean_squared_error(&y_pred, &y)?;
        model.clear_grads();
        loss.backward()?;
        optimizer.update()?;

        if i % (iters / 10) == 0 || i == iters - 1 {
            println!("{} {:.10}", i, loss);
        }
    }

    match create_dir("output") {
        Ok(_) => println!("create output directory"),
        Err(_) => {},
    }

    let data: Vec<(f64, f64)> = x_data.iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let x_max = data.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let x_min = data.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);

    let line_x: Vec<f64> = (0..=100)
        .map(|x| x as f64 / 100.0 * (x_max - x_min) + x_min)
        .collect();

    let model = optimizer.get_model_mut_result()?;
    let y = model.forward(
        &[Variable::new(
                Tensor::new(
                        line_x.clone(),
                        vec![line_x.len(), 1])?.into())]
    )?.remove(0);
    let line_y = y.data().to_f64_tensor()?.get_data().clone();
    let line_points = line_x.iter()
        .zip(line_y.iter())
        .map(|(&x, &y)| (x, y)).collect();

    plot_data_and_line(
        &data,
        "output/step46.png",
        line_points
    )?;

    Ok(())
}

#[test]
fn step47() -> Result<()> {
    use ktensor::Tensor;
    use kdezero::Variable;
    use kdezero::function::{
        get_item_with_one_index,
        get_item_with_one_indexes,
        get_item_with_indexes,
    };

    let x = Variable::new(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3])?.into());
    let mut y = get_item_with_one_index(&x, 1)?;
    println!("{}", y);
    assert_eq!(
        *y.data(),
        Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?.into());

    y.backward()?;
    println!("{}", x.grad_result()?);
    assert_eq!(
        *x.grad_result()?.data(),
        Tensor::new(vec![0., 0., 0., 1., 1., 1.], vec![2, 3])?.into());

    let indices = vec![0, 0, 1];
    let y = get_item_with_one_indexes(&x, &indices)?;
    println!("{}", y);
    assert_eq!(
        *y.data(),
        Tensor::new(vec![1., 2., 3., 1., 2., 3., 4., 5., 6.], vec![3, 3])?.into());

    let indices = vec![vec![0, 1], vec![2, 2]];
    let y = get_item_with_indexes(&x, &indices)?;
    println!("{}", y);
    assert_eq!(
        *y.data(),
        Tensor::new(vec![3., 6.], vec![2])?.into());

    Ok(())
}

#[test]
fn step48() -> Result<()> {
    use ktensor::tensor::TensorRng;
    use kdezero::{Variable, Model, Optimizer};
    use kdezero::function::{softmax_cross_entropy, sigmoid};
    use kdezero::model::MLP;
    use kdezero::optimizer::SGD;
    use kdezero::data_set::sample::get_spiral;

    let layer = MLP::new(&[2, 10, 3], sigmoid)?;
    let model = Model::new(layer);
    let opt_content = SGD::new(1.0);
    let mut optimizer = Optimizer::new(opt_content);
    optimizer.set_model(model);

    let (x, t) = get_spiral(true)?;

    let max_epoch = 300;
    let batch_size = 30;
    let data_size = x.len();
    let max_iter = (data_size - 1) / batch_size + 1;

    let mut rng = TensorRng::new();

    for epoch in 0..max_epoch {
        let batch_index = rng.permutation(data_size);
        let mut sum_loss = 0.0;

        for i in 0..max_iter {
            let batch_start = i * batch_size;
            let batch_end = std::cmp::min(batch_start + batch_size, data_size);
            let batch_index = batch_index[batch_start..batch_end]
                .iter().map(|&i| i as usize).collect::<Vec<_>>();
            let batch_x: Variable = x
                .slice_with_one_indexes(&batch_index)?
                .into();
            let batch_t: Variable = t
                .slice_with_one_indexes(&batch_index)?
                .into();

            let model = optimizer.get_model_mut_result()?;
            let y = model.forward(&[batch_x])?.remove(0);
            let mut loss = softmax_cross_entropy(&y, &batch_t)?;
            model.clear_grads();
            loss.backward()?;
            optimizer.update()?;

            sum_loss += *loss.data()
                .to_f64_tensor()?.get_data().get(0).unwrap()
                * batch_size as f64;
        }

        let avg_loss = sum_loss / data_size as f64;
        if epoch % 10 == 0 || epoch == max_epoch - 1 {
            println!("epoch {} loss {:.10}", epoch, avg_loss);
        }
    }

    let (x, t) = get_spiral(false)?;
    let model = optimizer.get_model_mut_result()?;
    let y = model.forward(&[x.into()])?.remove(0);
    let y = y.data().to_f64_tensor()?
        .argmax_with_axis(1, false)?;
    let sum_true = y.iter()
        .zip(t.iter())
        .map(|(&y, &t)| if y == t { 1 } else { 0 })
        .sum::<usize>();
    let accuracy = sum_true as f64 / y.get_shape()[0] as f64;
    println!("accuracy: {:.4}", accuracy);
    Ok(())
}

#[test]
fn step49() -> Result<()> {
    use ktensor::tensor::{Tensor, TensorRng};
    use kdezero::{Variable, Model, Optimizer};
    use kdezero::function::{softmax_cross_entropy, sigmoid};
    use kdezero::model::MLP;
    use kdezero::optimizer::SGD;
    use kdezero::data_set::{DataSet, sample::Spiral};

    let layer = MLP::new(&[2, 10, 3], sigmoid)?;
    let model = Model::new(layer);
    let opt_content = SGD::new(1.0);
    let mut optimizer = Optimizer::new(opt_content);
    optimizer.set_model(model);

    let train_set = Spiral::new(true)?;

    let max_epoch = 300;
    let batch_size = 30;
    let data_size = train_set.len()?;
    let max_iter = (data_size - 1) / batch_size + 1;

    let mut rng = TensorRng::new();

    for epoch in 0..max_epoch {
        let batch_index = rng.permutation(data_size);
        let mut sum_loss = 0.0;

        for i in 0..max_iter {
            let batch_start = i * batch_size;
            let batch_end = std::cmp::min(batch_start + batch_size, data_size);
            let batch_index = batch_index[batch_start..batch_end]
                .iter().map(|&i| i as usize).collect::<Vec<_>>();
            let mut batch_x = vec![];
            let mut batch_t = vec![];
            for i in batch_index {
                let (x, t) = train_set.get(i)?;
                batch_x.push(x);
                batch_t.push(t.unwrap());
            }
            let batch_x: Variable = Tensor::from_tensor_list(&batch_x
                .iter()
                .map(|x| x)
                .collect::<Vec<_>>())?.into();
            let batch_t: Variable = Tensor::from_tensor_list(&batch_t
                .iter()
                .map(|t| t)
                .collect::<Vec<_>>())?.into();

            let model = optimizer.get_model_mut_result()?;
            let y = model.forward(&[batch_x])?.remove(0);
            let mut loss = softmax_cross_entropy(&y, &batch_t)?;
            model.clear_grads();
            loss.backward()?;
            optimizer.update()?;

            sum_loss += *loss.data()
                .to_f64_tensor()?.get_data().get(0).unwrap()
                * batch_size as f64;
        }

        let avg_loss = sum_loss / data_size as f64;
        if epoch % 10 == 0 || epoch == max_epoch - 1 {
            println!("epoch {} loss {:.10}", epoch, avg_loss);
        }
    }

    let test_set = Spiral::new(false)?;
    let x = test_set.get_all_data()?.unwrap().clone().into();
    let t = test_set.get_all_label()?.unwrap();
    let model = optimizer.get_model_mut_result()?;
    let y = model.forward(&[x])?.remove(0);
    let y = y.data().to_f64_tensor()?
        .argmax_with_axis(1, false)?;
    let sum_true = y.iter()
        .zip(t.iter())
        .map(|(&y, &t)| if y == t { 1 } else { 0 })
        .sum::<usize>();
    let accuracy = sum_true as f64 / y.get_shape()[0] as f64;
    println!("accuracy: {:.4}", accuracy);
    Ok(())
}
