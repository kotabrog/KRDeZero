use anyhow::Result;

#[test]
fn step1() {
    use kdezero::Variable;

    let data = 1.0;
    let mut x = Variable::from(data);
    println!("{:?}", x.data());

    assert_eq!(x.data(), &1.0.into());

    x.set_data(2.0.into());
    println!("{:?}", x.data());

    assert_eq!(x.data(), &2.0.into());
}

#[test]
fn step2() -> Result<()> {
    use kdezero::{Variable, Function};
    use kdezero::function::Square;

    let x = Variable::from(10);
    let f = Square::new();
    let y = f.forward(&vec![x])?;

    println!("{:?}", y[0].data());

    assert_eq!(y[0].data(), &100.into());
    Ok(())
}

#[test]
fn step3() -> Result<()> {
    use kdezero::{Variable, Function};
    use kdezero::function::{Square, Exp};

    let x = Variable::from(0.5);
    let a = Square::new();
    let b = Exp::new();
    let c = Square::new();

    let y1 = a.forward(&vec![x])?;
    let y2 = b.forward(&y1)?;
    let y3 = c.forward(&y2)?;

    println!("{:?}", y3[0].data());
    assert_eq!(y3[0].data(), &0.5f32.powi(2).exp().powi(2).into());

    Ok(())
}
