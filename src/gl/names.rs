use std::collections::BTreeMap;

use super::{GlError, GlResult};

/// A GL name space: glGen* reserves names, the first bind creates the object, glDelete* frees both.
/// Name 0 is never an object (the default binding).
pub struct Names<T> {
    next: u32,
    reserved: BTreeMap<u32, Option<T>>,
}

impl<T> Default for Names<T> {
    fn default() -> Self {
        Names { next: 1, reserved: BTreeMap::new() }
    }
}

impl<T> Names<T> {
    pub fn reserve_n(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.reserve(None)).collect()
    }

    /// glCreate*: a name with its object
    pub fn create(&mut self, object: T) -> u32 {
        self.reserve(Some(object))
    }

    fn reserve(&mut self, object: Option<T>) -> u32 {
        while self.reserved.contains_key(&self.next) || self.next == 0 {
            self.next = self.next.wrapping_add(1);
        }
        let name = self.next;
        self.reserved.insert(name, object);
        self.next = self.next.wrapping_add(1);
        name
    }

    /// glBind* on buffers, textures, renderbuffers and framebuffers: any name, generated or not, has its object
    /// made by the first bind (GL ES 3.0 keeps ES 2.0's rule)
    pub fn bind_any(&mut self, name: u32, make: impl FnOnce() -> T) {
        self.reserved.entry(name).or_insert(None).get_or_insert_with(make);
    }

    /// glBindVertexArray: a name glGen* gave creates its object; any other name is an error
    pub fn bind(&mut self, name: u32, make: impl FnOnce() -> T) -> GlResult<()> {
        match self.reserved.get_mut(&name) {
            Some(slot) => {
                slot.get_or_insert_with(make);
                Ok(())
            }
            None => Err(GlError::InvalidOperation),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.reserved.iter().filter_map(|(n, o)| o.as_ref().map(|o| (*n, o)))
    }

    pub fn delete(&mut self, name: u32) -> Option<T> {
        self.reserved.remove(&name).flatten()
    }

    /// glIs*: only a name with an object
    pub fn is(&self, name: u32) -> bool {
        self.reserved.get(&name).is_some_and(Option::is_some)
    }

    pub fn get(&self, name: u32) -> Option<&T> {
        self.reserved.get(&name).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, name: u32) -> Option<&mut T> {
        self.reserved.get_mut(&name).and_then(Option::as_mut)
    }
}
