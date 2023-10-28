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
