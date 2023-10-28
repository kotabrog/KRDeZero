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
fn step2() -> Result<()>{
    use kdezero::{Variable, Function};
    use kdezero::function::operator::Square;

    let x = Variable::from(10);
    let f = Square::new();
    let y = f.forward(vec![x])?;

    println!("{:?}", y[0].data());

    assert_eq!(y[0].data(), &100.into());
    Ok(())
}
