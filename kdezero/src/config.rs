use std::cell::RefCell;

thread_local! {
    pub static NO_GRAD: RefCell<bool> = RefCell::new(false);
}

pub struct NoGradGuard;

impl NoGradGuard {
    pub fn new() -> Self {
        NO_GRAD.with(|no_grad| {
            *no_grad.borrow_mut() = true;
        });
        Self
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        NO_GRAD.with(|no_grad| {
            *no_grad.borrow_mut() = false;
        });
    }
}

pub fn no_grad() -> NoGradGuard {
    NoGradGuard::new()
}

pub fn is_no_grad_enabled() -> bool {
    NO_GRAD.with(|no_grad| *no_grad.borrow())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_grad_normal() {
        assert!(!is_no_grad_enabled());
        {
            let _guard = no_grad();
            assert!(is_no_grad_enabled());
        }
        assert!(!is_no_grad_enabled());
    }
}
