use std::collections::{BinaryHeap, HashSet};
use anyhow::Result;
use crate::Function;
use crate::function::add;
use super::Variable;

struct OrdFunction {
    pub function: Function,
    pub generation: usize,
}

impl PartialEq for OrdFunction {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
    }
}

impl Eq for OrdFunction {}

impl PartialOrd for OrdFunction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.generation.partial_cmp(&other.generation)
    }
}

impl Ord for OrdFunction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.generation.cmp(&other.generation)
    }
}

fn add_func(func: &Function, funcs: &mut BinaryHeap<OrdFunction>, seen_set: &mut HashSet<Function>) {
    if seen_set.contains(func) {
        return;
    }
    seen_set.insert(func.clone());
    let generation = func.generation();
    funcs.push(OrdFunction { function: func.clone(), generation });
}

impl Variable {
    fn backward_inner(&mut self, retain_grad: bool) -> Result<()> {
        self.set_default_grad_if_none()?;

        let mut funcs = BinaryHeap::new();
        let mut seen_set = HashSet::new();

        add_func(&self.get_creator_clone_result()?, &mut funcs, &mut seen_set);

        while !funcs.is_empty() {
            let f = funcs.pop().unwrap()
                .function;
            let xs = f.inputs_clone_result()?;
            let ys = f.outputs_clone_result()?;
            let grad = ys
                .iter()
                .map(|y| y.grad_result())
                .collect::<Result<Vec<_>>>()?;
            let xgs = f.backward(&grad)?;
            for (mut x, xg) in xs.into_iter().zip(xgs) {
                if x.is_grad_none() {
                    x.set_grad(xg);
                } else {
                    x.set_grad(add(&x.grad_result()?, &xg)?);
                }
                if let Some(c) = x.get_creator_clone() {
                    add_func(&c, &mut funcs, &mut seen_set);
                }
            }
            if !retain_grad {
                for mut y in f.outputs_clone_result()? {
                    y.clear_grad();
                }
            }
        }
        Ok(())
    }

    pub fn backward(&mut self) -> Result<()> {
        self.backward_inner(false)
    }

    pub fn backward_retain_grad(&mut self) -> Result<()> {
        self.backward_inner(true)
    }
}
