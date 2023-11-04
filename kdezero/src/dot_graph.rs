use std::fs::File;
use std::io::prelude::*;
use std::process::Command;
use anyhow::Result;
use std::collections::{HashSet, VecDeque};
use super::Variable;
use super::Function;

fn variable_to_dot(v: &Variable, verbose: bool) -> String {
    let mut label = format!("{}", v.name());
    if verbose {
        label += &format!(": {:?} {}", v.shape(), v.data_type());
    }

    format!("{} [label=\"{}\", color=orange, style=filled]\n", v.id(), label)
}

fn function_to_dot(f: &Function) -> String {
    let mut txt = format!("{} [label=\"{}\", color=lightblue, style=filled, shape=box]\n",
        f.id(), f.function_name());

    if let Ok(inputs) = f.inputs_clone_result() {
        for input in inputs {
            txt += &format!("{} -> {}\n", input.id(), f.id());
        }
    }
    if let Ok(outputs) = f.outputs_clone_result() {
        for output in outputs {
            txt += &format!("{} -> {}\n", f.id(), output.id());
        }
    }

    txt
}

pub fn get_dot_graph(output: &Variable, verbose: bool) -> Result<String> {
    let mut txt = String::new();
    let mut funcs = VecDeque::new();
    let mut seen_set = HashSet::new();

    let func = output.get_creator_clone_result()?;
    funcs.push_back(func.clone());
    seen_set.insert(func);
    txt += &variable_to_dot(output, verbose);

    while !funcs.is_empty() {
        let f = funcs.pop_front().unwrap();
        txt += &function_to_dot(&f);
        for input in f.inputs_clone_result()? {
            txt += &variable_to_dot(&input, verbose);
            if let Ok(func) = input.get_creator_clone_result() {
                if !seen_set.contains(&func) {
                    funcs.push_back(func.clone());
                    seen_set.insert(func);
                }
            }
        }
    }

    Ok(format!("digraph g {{\n{}}}", txt))
}

fn write_dot_graph_to_file(output: &Variable, out_path: &str, verbose: bool) -> Result<()> {
    let text = get_dot_graph(output, verbose)?;
    let mut file = File::create(out_path)?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

pub fn plot_dot_graph(output: &Variable, out_path_without_extension: &str, to_png: bool, verbose: bool) -> Result<()> {
    let dot_path = format!("{}.dot", out_path_without_extension);
    write_dot_graph_to_file(output, &dot_path, verbose)?;
    if to_png {
        let png_path = format!("{}.png", out_path_without_extension);
        Command::new("dot")
            .args([dot_path.as_str(), "-T", "png", "-o", png_path.as_str()])
            .status()?;
    }
    Ok(())
}
