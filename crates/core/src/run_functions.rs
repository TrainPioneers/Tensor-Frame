#[derive(Debug, Hash, PartialEq, Eq)]
pub(crate) enum RunOperation {
    Add,
    Sub,
    Mul,
    Div,
    Dot
}