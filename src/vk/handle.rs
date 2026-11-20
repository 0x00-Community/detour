use core::fmt;
use core::marker::PhantomData;

use super::VkError;

/// A Vulkan object's handle: which object of its kind `K` (handles are not reused, so a destroyed object's handle
/// stays invalid)
pub struct VkHandle<K>(u32, PhantomData<fn() -> K>);

impl<K> Clone for VkHandle<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for VkHandle<K> {}

impl<K> PartialEq for VkHandle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K> Eq for VkHandle<K> {}

impl<K> fmt::Debug for VkHandle<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", core::any::type_name::<K>().rsplit("::").next().unwrap_or("object"), self.0)
    }
}

/// The objects of one kind a device holds, by handle
pub struct VkObjects<T, K = T> {
    items: Vec<Option<T>>,
    kind: PhantomData<fn() -> K>,
}

impl<T, K> Default for VkObjects<T, K> {
    fn default() -> Self {
        VkObjects { items: Vec::new(), kind: PhantomData }
    }
}

impl<T, K> VkObjects<T, K> {
    pub fn insert(&mut self, t: T) -> VkHandle<K> {
        self.items.push(Some(t));
        VkHandle((self.items.len() - 1) as u32, PhantomData)
    }

    pub fn get(&self, h: VkHandle<K>) -> Result<&T, VkError> {
        self.items.get(h.0 as usize).and_then(Option::as_ref).ok_or_else(|| VkError::ValidationFailed(format!("{h:?}: not a live handle")))
    }

    pub fn get_mut(&mut self, h: VkHandle<K>) -> Result<&mut T, VkError> {
        self.items.get_mut(h.0 as usize).and_then(Option::as_mut).ok_or_else(|| VkError::ValidationFailed(format!("{h:?}: not a live handle")))
    }

    /// vkDestroy*: destroying a handle that is not live does nothing (as VK_NULL_HANDLE does)
    pub fn remove(&mut self, h: VkHandle<K>) -> Option<T> {
        self.items.get_mut(h.0 as usize).and_then(Option::take)
    }
}
