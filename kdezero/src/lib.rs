pub mod error;
mod variable;
pub mod function;
pub mod test_utility;
mod config;
mod dot_graph;
pub mod layer;
mod utility;

pub use variable::{Variable, VariableData, VariableWeak, VariableType};
pub use function::{Function, FunctionInner, FunctionContent};
pub use layer::{Layer, LayerInner, LayerContent};
pub use config::{no_grad, no_grad_frag, is_no_grad_enabled};
pub use dot_graph::{get_dot_graph, plot_dot_graph};
