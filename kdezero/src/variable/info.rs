use std::rc::Rc;
use std::cell::Ref;
use super::Variable;

impl Variable {
    pub fn shape(&self) -> Ref<[usize]> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| inner.data.shape())
    }

    pub fn ndim(&self) -> usize {
        let inner = self.inner.borrow();
        inner.data.ndim()
    }

    pub fn size(&self) -> usize {
        let inner = self.inner.borrow();
        inner.data.size()
    }

    pub fn len(&self) -> usize {
        let inner = self.inner.borrow();
        inner.data.len()
    }

    pub fn data_type(&self) -> Ref<str> {
        let inner = self.inner.borrow();
        Ref::map(inner, |inner| inner.data.data_type())
    }

    pub fn is_none(&self) -> bool {
        let inner = self.inner.borrow();
        inner.data.is_none()
    }

    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }
}
