// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(missing_docs)]
#![feature(try_from)]

//! A simple map based on a vector for small integer keys. Space requirements
//! are O(highest integer key).

// optional serde support
#[cfg(feature = "eders")]
extern crate serde;
#[cfg(feature = "eders")]
#[macro_use]
extern crate serde_derive;

use self::Entry::*;

use std::cmp::{max, Ordering};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Enumerate, FilterMap, FromIterator};
use std::marker::PhantomData;
use std::mem::{replace, swap};
use std::ops::{Index, IndexMut};
use std::slice;
use std::vec;

/// Types that implement `Key` can be converted to and from `usize` which enables them to be used as
/// the key type for `VecMap`s.
pub trait Key: Copy + Eq {
    /// Convert an instance of Key into a usize.
    fn into_usize(&self) -> usize;
    /// Convert a usize into an instance of Key.
    fn from_usize(val: usize) -> Self;
}
impl Key for usize {
    fn into_usize(&self) -> usize {
        *self
    }
    fn from_usize(val: usize) -> Self {
        val
    }
}

/// A map optimized for small integer keys.
///
/// # Examples
///
/// ```
/// use vec_map::VecMap;
///
/// let mut months = VecMap::<usize, _>::new();
/// months.insert(1, "Jan");
/// months.insert(2, "Feb");
/// months.insert(3, "Mar");
///
/// if !months.contains_key(12) {
///     println!("The end is near!");
/// }
///
/// assert_eq!(months.get(1), Some(&"Jan"));
///
/// if let Some(value) = months.get_mut(3) {
///     *value = "Venus";
/// }
///
/// assert_eq!(months.get(3), Some(&"Venus"));
///
/// // Print out all months
/// for (key, value) in &months {
///     println!("month {} is {}", key, value);
/// }
///
/// months.clear();
/// assert!(months.is_empty());
/// ```
#[cfg_attr(feature = "eders", derive(Serialize, Deserialize))]
#[derive(Clone, Default)]
pub struct VecMap<K: Key, V> {
    n: usize,
    v: Vec<Option<V>>,
    phantom: PhantomData<K>,
}

/// A view into a single entry in a map, which may either be vacant or occupied.
pub enum Entry<'a, K: 'a + Key, V: 'a> {
    /// A vacant Entry
    Vacant(VacantEntry<'a, K, V>),

    /// An occupied Entry
    Occupied(OccupiedEntry<'a, K, V>),
}

/// A vacant Entry.
pub struct VacantEntry<'a, K: 'a + Key, V: 'a> {
    map: &'a mut VecMap<K, V>,
    index: usize,
}

/// An occupied Entry.
pub struct OccupiedEntry<'a, K: 'a + Key, V: 'a> {
    map: &'a mut VecMap<K, V>,
    index: usize,
}

impl<K: Key + Hash, V: Hash> Hash for VecMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // In order to not traverse the `VecMap` twice, count the elements
        // during iteration.
        let mut count: usize = 0;
        for elt in self {
            elt.hash(state);
            count += 1;
        }
        count.hash(state);
    }
}

impl<K: Key, V> VecMap<K, V> {
    /// Creates an empty `VecMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    /// let mut map: VecMap<usize, &str> = VecMap::new();
    /// ```
    pub fn new() -> Self {
        VecMap {
            n: 0,
            v: vec![],
            phantom: PhantomData,
        }
    }

    /// Creates an empty `VecMap` with space for at least `capacity`
    /// elements before resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    /// let mut map: VecMap<usize, &str> = VecMap::with_capacity(10);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        VecMap {
            n: 0,
            v: Vec::with_capacity(capacity),
            phantom: PhantomData,
        }
    }

    /// Returns the number of elements the `VecMap` can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    /// let map: VecMap<usize, String> = VecMap::with_capacity(10);
    /// assert!(map.capacity() >= 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.v.capacity()
    }

    /// Reserves capacity for the given `VecMap` to contain `len` distinct keys.
    /// In the case of `VecMap` this means reallocations will not occur as long
    /// as all inserted keys are less than `len`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    /// let mut map: VecMap<usize, &str> = VecMap::new();
    /// map.reserve_len(10);
    /// assert!(map.capacity() >= 10);
    /// ```
    pub fn reserve_len(&mut self, len: usize) {
        let cur_len = self.v.len();
        if len >= cur_len {
            self.v.reserve(len - cur_len);
        }
    }

    /// Reserves the minimum capacity for the given `VecMap` to contain `len` distinct keys.
    /// In the case of `VecMap` this means reallocations will not occur as long as all inserted
    /// keys are less than `len`.
    ///
    /// Note that the allocator may give the collection more space than it requests.
    /// Therefore capacity cannot be relied upon to be precisely minimal.  Prefer
    /// `reserve_len` if future insertions are expected.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    /// let mut map: VecMap<usize, &str> = VecMap::new();
    /// map.reserve_len_exact(10);
    /// assert!(map.capacity() >= 10);
    /// ```
    pub fn reserve_len_exact(&mut self, len: usize) {
        let cur_len = self.v.len();
        if len >= cur_len {
            self.v.reserve_exact(len - cur_len);
        }
    }

    /// Trims the `VecMap` of any excess capacity.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    /// let mut map: VecMap<usize, &str> = VecMap::with_capacity(10);
    /// map.shrink_to_fit();
    /// assert_eq!(map.capacity(), 0);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        // strip off trailing `None`s
        if let Some(idx) = self.v.iter().rposition(Option::is_some) {
            self.v.truncate(idx + 1);
        } else {
            self.v.clear();
        }

        self.v.shrink_to_fit()
    }

    /// Returns an iterator visiting all keys in ascending order of the keys.
    /// The iterator's element type is `K`.
    pub fn keys(&self) -> Keys<K, V> {
        Keys { iter: self.iter() }
    }

    /// Returns an iterator visiting all values in ascending order of the keys.
    /// The iterator's element type is `&'r V`.
    pub fn values(&self) -> Values<K, V> {
        Values { iter: self.iter() }
    }

    /// Returns an iterator visiting all values in ascending order of the keys.
    /// The iterator's element type is `&'r mut V`.
    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        ValuesMut {
            iter_mut: self.iter_mut(),
        }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order of the keys.
    /// The iterator's element type is `(K, &'r V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// // Print `1: a` then `2: b` then `3: c`
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            front: 0,
            back: self.v.len(),
            n: self.n,
            yielded: 0,
            iter: self.v.iter(),
            phantom: PhantomData,
        }
    }

    /// Returns an iterator visiting all key-value pairs in ascending order of the keys,
    /// with mutable references to the values.
    /// The iterator's element type is `(K, &'r mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.insert(3, "c");
    ///
    /// for (key, value) in map.iter_mut() {
    ///     *value = "x";
    /// }
    ///
    /// for (key, value) in &map {
    ///     assert_eq!(value, &"x");
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            front: 0,
            back: self.v.len(),
            n: self.n,
            yielded: 0,
            iter: self.v.iter_mut(),
            phantom: PhantomData,
        }
    }

    /// Moves all elements from `other` into the map while overwriting existing keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut a = VecMap::<usize, _>::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// let mut b = VecMap::<usize, _>::new();
    /// b.insert(3, "c");
    /// b.insert(4, "d");
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(b.len(), 0);
    /// assert_eq!(a[1], "a");
    /// assert_eq!(a[2], "b");
    /// assert_eq!(a[3], "c");
    /// assert_eq!(a[4], "d");
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        self.extend(other.drain());
    }

    /// Splits the collection into two at the given key.
    ///
    /// Returns a newly allocated `Self`. `self` contains elements `[0, at)`,
    /// and the returned `Self` contains elements `[at, max_key)`.
    ///
    /// Note that the capacity of `self` does not change.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut a = VecMap::<usize, _>::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    /// a.insert(3, "c");
    /// a.insert(4, "d");
    ///
    /// let b = a.split_off(3);
    ///
    /// assert_eq!(a[1], "a");
    /// assert_eq!(a[2], "b");
    ///
    /// assert_eq!(b[3], "c");
    /// assert_eq!(b[4], "d");
    /// ```
    pub fn split_off(&mut self, at: usize) -> Self {
        let mut other = VecMap::new();

        if at == 0 {
            // Move all elements to other
            // The swap will also fix .n
            swap(self, &mut other);
            return other;
        } else if at >= self.v.len() {
            // No elements to copy
            return other;
        }

        // Look up the index of the first non-None item
        let first_index = self.v.iter().position(|el| el.is_some());
        let start_index = match first_index {
            Some(index) => max(at, index),
            None => {
                // self has no elements
                return other;
            }
        };

        // Fill the new VecMap with `None`s until `start_index`
        other.v.extend((0..start_index).map(|_| None));

        // Move elements beginning with `start_index` from `self` into `other`
        let mut taken = 0;
        other.v.extend(self.v[start_index..].iter_mut().map(|el| {
            let el = el.take();
            if el.is_some() {
                taken += 1;
            }
            el
        }));
        other.n = taken;
        self.n -= taken;

        other
    }

    /// Returns an iterator visiting all key-value pairs in ascending order of
    /// the keys, emptying (but not consuming) the original `VecMap`.
    /// The iterator's element type is `(K, &'r V)`. Keeps the allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// let vec: Vec<(usize, &str)> = map.drain().collect();
    ///
    /// assert_eq!(vec, [(1, "a"), (2, "b"), (3, "c")]);
    /// ```
    pub fn drain(&mut self) -> Drain<K, V> {
        fn filter<K: Key, A>((i, v): (usize, Option<A>)) -> Option<(K, A)> {
            v.map(|v| (K::from_usize(i), v))
        }
        let filter: fn((usize, Option<V>)) -> Option<(K, V)> = filter; // coerce to fn ptr

        self.n = 0;
        Drain {
            iter: self.v.drain(..).enumerate().filter_map(filter),
        }
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut a = VecMap::<usize, _>::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut a = VecMap::<usize, _>::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut a = VecMap::<usize, _>::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.n = 0;
        self.v.clear()
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(1), Some(&"a"));
    /// assert_eq!(map.get(2), None);
    /// ```
    pub fn get(&self, key: K) -> Option<&V> {
        let key = key.into_usize();
        if key < self.v.len() {
            match self.v[key] {
                Some(ref value) => Some(value),
                None => None,
            }
        } else {
            None
        }
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(1), true);
    /// assert_eq!(map.contains_key(2), false);
    /// ```
    #[inline]
    pub fn contains_key(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[1], "b");
    /// ```
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let key = key.into_usize();
        if key < self.v.len() {
            match *(&mut self.v[key]) {
                Some(ref mut value) => Some(value),
                None => None,
            }
        } else {
            None
        }
    }

    /// Inserts a key-value pair into the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[37], "c");
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let key = key.into_usize();
        let len = self.v.len();
        if len <= key {
            self.v.extend((0..key - len + 1).map(|_| None));
        }
        let was = replace(&mut self.v[key], Some(value));
        if was.is_none() {
            self.n += 1;
        }
        was
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(1), Some("a"));
    /// assert_eq!(map.remove(1), None);
    /// ```
    pub fn remove(&mut self, key: K) -> Option<V> {
        let key = key.into_usize();
        if key >= self.v.len() {
            return None;
        }
        let result = &mut self.v[key];
        let was = result.take();
        if was.is_some() {
            self.n -= 1;
        }
        was
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut count: VecMap<usize, u32> = VecMap::<usize, _>::new();
    ///
    /// // count the number of occurrences of numbers in the vec
    /// for x in vec![1, 2, 1, 2, 3, 4, 1, 2, 4] {
    ///     *count.entry(x).or_insert(0) += 1;
    /// }
    ///
    /// assert_eq!(count[1], 3);
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        // FIXME(Gankro): this is basically the dumbest implementation of
        // entry possible, because weird non-lexical borrows issues make it
        // completely insane to do any other way. That said, Entry is a border-line
        // useless construct on VecMap, so it's hardly a big loss.
        if self.contains_key(key) {
            Occupied(OccupiedEntry {
                map: self,
                index: key.into_usize(),
            })
        } else {
            Vacant(VacantEntry {
                map: self,
                index: key.into_usize(),
            })
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` such that `f(&k, &mut v)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map: VecMap<usize, usize> = (0..8).map(|x|(x, x*10)).collect();
    /// map.retain(|k, _| k % 2 == 0);
    /// assert_eq!(map.len(), 4);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(K, &mut V) -> bool,
    {
        for (i, e) in self.v.iter_mut().enumerate() {
            let remove = match *e {
                Some(ref mut value) => !f(K::from_usize(i), value),
                None => false,
            };
            if remove {
                *e = None;
                self.n -= 1;
            }
        }
    }
}

impl<'a, K: Key, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty, and
    /// returns a mutable reference to the value in the entry.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default
    /// function if empty, and returns a mutable reference to the value in the
    /// entry.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default()),
        }
    }
}

impl<'a, K: Key, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        let index = K::from_usize(self.index);
        self.map.insert(index, value);
        &mut self.map[index]
    }
}

impl<'a, K: Key, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        let index = K::from_usize(self.index);
        &self.map[index]
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        let index = K::from_usize(self.index);
        &mut self.map[index]
    }

    /// Converts the entry into a mutable reference to its value.
    pub fn into_mut(self) -> &'a mut V {
        let index = K::from_usize(self.index);
        &mut self.map[index]
    }

    /// Sets the value of the entry with the OccupiedEntry's key,
    /// and returns the entry's old value.
    pub fn insert(&mut self, value: V) -> V {
        let index = K::from_usize(self.index);
        self.map.insert(index, value).unwrap()
    }

    /// Takes the value of the entry out of the map, and returns it.
    pub fn remove(self) -> V {
        let index = K::from_usize(self.index);
        self.map.remove(index).unwrap()
    }
}

impl<K: Key, V: PartialEq> PartialEq for VecMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.iter().eq(other.iter())
    }
}

impl<K: Key, V: Eq> Eq for VecMap<K, V> {}

impl<K: Key + PartialOrd, V: PartialOrd> PartialOrd for VecMap<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Key + Ord, V: Ord> Ord for VecMap<K, V> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K: Key + fmt::Debug, V: fmt::Debug> fmt::Debug for VecMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self).finish()
    }
}

impl<K: Key, V> FromIterator<(K, V)> for VecMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<K: Key, T> IntoIterator for VecMap<K, T> {
    type Item = (K, T);
    type IntoIter = IntoIter<K, T>;

    /// Returns an iterator visiting all key-value pairs in ascending order of
    /// the keys, consuming the original `VecMap`.
    /// The iterator's element type is `(K, &'r V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vec_map::VecMap;
    ///
    /// let mut map = VecMap::<usize, _>::new();
    /// map.insert(1, "a");
    /// map.insert(3, "c");
    /// map.insert(2, "b");
    ///
    /// let vec: Vec<(usize, &str)> = map.into_iter().collect();
    ///
    /// assert_eq!(vec, [(1, "a"), (2, "b"), (3, "c")]);
    /// ```
    fn into_iter(self) -> IntoIter<K, T> {
        IntoIter {
            n: self.n,
            yielded: 0,
            iter: self.v.into_iter().enumerate(),
            phantom: PhantomData,
        }
    }
}

impl<'a, K: Key, T> IntoIterator for &'a VecMap<K, T> {
    type Item = (K, &'a T);
    type IntoIter = Iter<'a, K, T>;

    fn into_iter(self) -> Iter<'a, K, T> {
        self.iter()
    }
}

impl<'a, K: Key, T> IntoIterator for &'a mut VecMap<K, T> {
    type Item = (K, &'a mut T);
    type IntoIter = IterMut<'a, K, T>;

    fn into_iter(self) -> IterMut<'a, K, T> {
        self.iter_mut()
    }
}

impl<K: Key, V> Extend<(K, V)> for VecMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K: Key, V: Copy> Extend<(K, &'a V)> for VecMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, &'a V)>>(&mut self, iter: I) {
        self.extend(iter.into_iter().map(|(key, &value)| (key, value)));
    }
}

impl<K: Key, V> Index<K> for VecMap<K, V> {
    type Output = V;

    #[inline]
    fn index(&self, i: K) -> &V {
        self.get(i).expect("key not present")
    }
}

impl<'a, K: Key, V> Index<&'a K> for VecMap<K, V> {
    type Output = V;

    #[inline]
    fn index(&self, i: &K) -> &V {
        self.get(*i).expect("key not present")
    }
}

impl<K: Key, V> IndexMut<K> for VecMap<K, V> {
    #[inline]
    fn index_mut(&mut self, i: K) -> &mut V {
        self.get_mut(i).expect("key not present")
    }
}

impl<'a, K: Key, V> IndexMut<&'a K> for VecMap<K, V> {
    #[inline]
    fn index_mut(&mut self, i: &K) -> &mut V {
        self.get_mut(*i).expect("key not present")
    }
}

macro_rules! iterator {
    (impl $name:ident -> (K, $val:ty), $($getter:ident),+) => {
        impl<'a, K: Key, V> Iterator for $name<'a, K, V> {
            type Item = (K, $val);

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                while self.front < self.back {
                    match self.iter.next() {
                        Some(elem) => {
                            match elem$(. $getter ())+ {
                                Some(x) => {
                                    let index = self.front;
                                    self.front += 1;
                                    self.yielded += 1;
                                    let k = K::from_usize(index);
                                    return Some((k, x));
                                },
                                None => {},
                            }
                        }
                        _ => ()
                    }
                    self.front += 1;
                }
                None
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.n - self.yielded, Some(self.n - self.yielded))
            }
        }
    }
}

macro_rules! double_ended_iterator {
    (impl $name:ident -> $elem:ty, $($getter:ident),+) => {
        impl<'a, K: Key, V> DoubleEndedIterator for $name<'a, K, V> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                while self.front < self.back {
                    match self.iter.next_back() {
                        Some(elem) => {
                            match elem$(. $getter ())+ {
                                Some(x) => {
                                    self.back -= 1;
                                    return Some((K::from_usize(self.back), x));
                                },
                                None => {},
                            }
                        }
                        _ => ()
                    }
                    self.back -= 1;
                }
                None
            }
        }
    }
}

/// An iterator over the key-value pairs of a map.
#[derive(Clone)]
pub struct Iter<'a, K: 'a + Key, V: 'a> {
    front: usize,
    back: usize,
    n: usize,
    yielded: usize,
    iter: slice::Iter<'a, Option<V>>,
    phantom: PhantomData<K>,
}

iterator! { impl Iter -> (K, &'a V), as_ref }
impl<'a, K: Key, V> ExactSizeIterator for Iter<'a, K, V> {}
double_ended_iterator! { impl Iter -> (K, &'a V), as_ref }

/// An iterator over the key-value pairs of a map, with the
/// values being mutable.
pub struct IterMut<'a, K: 'a + Key, V: 'a> {
    front: usize,
    back: usize,
    n: usize,
    yielded: usize,
    iter: slice::IterMut<'a, Option<V>>,
    phantom: PhantomData<K>,
}

iterator! { impl IterMut -> (K, &'a mut V), as_mut }
impl<'a, K: Key, V> ExactSizeIterator for IterMut<'a, K, V> {}
double_ended_iterator! { impl IterMut -> (K, &'a mut V), as_mut }

/// An iterator over the keys of a map.
#[derive(Clone)]
pub struct Keys<'a, K: 'a + Key, V: 'a> {
    iter: Iter<'a, K, V>,
}

/// An iterator over the values of a map.
#[derive(Clone)]
pub struct Values<'a, K: 'a + Key, V: 'a> {
    iter: Iter<'a, K, V>,
}

/// An iterator over the values of a map.
pub struct ValuesMut<'a, K: 'a + Key, V: 'a> {
    iter_mut: IterMut<'a, K, V>,
}

/// A consuming iterator over the key-value pairs of a map.
pub struct IntoIter<K: Key, V> {
    n: usize,
    yielded: usize,
    iter: Enumerate<vec::IntoIter<Option<V>>>,
    phantom: PhantomData<K>,
}

/// A draining iterator over the key-value pairs of a map.
pub struct Drain<'a, K: Key, V: 'a> {
    iter: FilterMap<Enumerate<vec::Drain<'a, Option<V>>>, fn((usize, Option<V>)) -> Option<(K, V)>>,
}

impl<'a, K: Key, V> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: Key, V> ExactSizeIterator for Drain<'a, K, V> {}

impl<'a, K: Key, V> DoubleEndedIterator for Drain<'a, K, V> {
    fn next_back(&mut self) -> Option<(K, V)> {
        self.iter.next_back()
    }
}

impl<'a, K: Key, V> Iterator for Keys<'a, K, V> {
    type Item = K;

    fn next(&mut self) -> Option<K> {
        self.iter.next().map(|e| e.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: Key, V> ExactSizeIterator for Keys<'a, K, V> {}

impl<'a, K: Key, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<K> {
        self.iter.next_back().map(|e| e.0)
    }
}

impl<'a, K: Key, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<(&'a V)> {
        self.iter.next().map(|e| e.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: Key, V> ExactSizeIterator for Values<'a, K, V> {}

impl<'a, K: Key, V> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a V)> {
        self.iter.next_back().map(|e| e.1)
    }
}

impl<'a, K: Key, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<(&'a mut V)> {
        self.iter_mut.next().map(|e| e.1)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter_mut.size_hint()
    }
}

impl<'a, K: Key, V> ExactSizeIterator for ValuesMut<'a, K, V> {}

impl<'a, K: Key, V> DoubleEndedIterator for ValuesMut<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a mut V> {
        self.iter_mut.next_back().map(|e| e.1)
    }
}

impl<K: Key, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        loop {
            match self.iter.next() {
                None => return None,
                Some((i, Some(value))) => {
                    self.yielded += 1;
                    return Some((K::from_usize(i), value));
                }
                _ => {}
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.n - self.yielded, Some(self.n - self.yielded))
    }
}

impl<K: Key, V> ExactSizeIterator for IntoIter<K, V> {}

impl<K: Key, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<(K, V)> {
        loop {
            match self.iter.next_back() {
                None => return None,
                Some((i, Some(value))) => return Some((K::from_usize(i), value)),
                _ => {}
            }
        }
    }
}

#[allow(dead_code)]
fn assert_properties() {
    fn vec_map_covariant<'a, K: Key, T>(map: VecMap<K, &'static T>) -> VecMap<K, &'a T> {
        map
    }

    fn into_iter_covariant<'a, K: Key, T>(iter: IntoIter<K, &'static T>) -> IntoIter<K, &'a T> {
        iter
    }

    fn iter_covariant<'i, 'a, K: Key, T>(iter: Iter<'i, K, &'static T>) -> Iter<'i, K, &'a T> {
        iter
    }

    fn keys_covariant<'i, 'a, K: Key, T>(iter: Keys<'i, K, &'static T>) -> Keys<'i, K, &'a T> {
        iter
    }

    fn values_covariant<'i, 'a, K: Key, T>(
        iter: Values<'i, K, &'static T>,
    ) -> Values<'i, K, &'a T> {
        iter
    }
}

#[cfg(test)]
mod test {
    use super::VecMap;
    use super::Entry::{Occupied, Vacant};
    use std::hash::{Hash, Hasher, SipHasher};

    #[test]
    fn test_get_mut() {
        let mut m = VecMap::<usize, _>::new();
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(5) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.get(5), Some(&new));
    }

    #[test]
    fn test_len() {
        let mut map = VecMap::<usize, _>::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.insert(5, 20).is_none());
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        assert!(map.insert(11, 12).is_none());
        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());
        assert!(map.insert(14, 22).is_none());
        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut map = VecMap::<usize, _>::new();
        assert!(map.insert(5, 20).is_none());
        assert!(map.insert(11, 12).is_none());
        assert!(map.insert(14, 22).is_none());
        map.clear();
        assert!(map.is_empty());
        assert!(map.get(5).is_none());
        assert!(map.get(11).is_none());
        assert!(map.get(14).is_none());
    }

    #[test]
    fn test_insert() {
        let mut m = VecMap::<usize, _>::new();
        assert_eq!(m.insert(1, 2), None);
        assert_eq!(m.insert(1, 3), Some(2));
        assert_eq!(m.insert(1, 4), Some(3));
    }

    #[test]
    fn test_remove() {
        let mut m = VecMap::<usize, _>::new();
        m.insert(1, 2);
        assert_eq!(m.remove(1), Some(2));
        assert_eq!(m.remove(1), None);
    }

    #[test]
    fn test_keys() {
        let mut map = VecMap::<usize, _>::new();
        map.insert(1, 'a');
        map.insert(2, 'b');
        map.insert(3, 'c');
        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let mut map = VecMap::<usize, _>::new();
        map.insert(1, 'a');
        map.insert(2, 'b');
        map.insert(3, 'c');
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_iterator() {
        let mut m = VecMap::<usize, _>::new();

        assert!(m.insert(0, 1).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        let mut it = m.iter();
        assert_eq!(it.size_hint(), (5, Some(5)));
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.size_hint(), (4, Some(4)));
        assert_eq!(it.next().unwrap(), (1, &2));
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.next().unwrap(), (3, &5));
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.next().unwrap(), (6, &10));
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next().unwrap(), (10, &11));
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_iterator_size_hints() {
        let mut m = VecMap::<usize, _>::new();

        assert!(m.insert(0, 1).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        assert_eq!(m.iter().size_hint(), (5, Some(5)));
        assert_eq!(m.iter().rev().size_hint(), (5, Some(5)));
        assert_eq!(m.iter_mut().size_hint(), (5, Some(5)));
        assert_eq!(m.iter_mut().rev().size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_mut_iterator() {
        let mut m = VecMap::<usize, _>::new();

        assert!(m.insert(0, 1).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        for (k, v) in &mut m {
            *v += k as isize;
        }

        let mut it = m.iter();
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.next().unwrap(), (1, &3));
        assert_eq!(it.next().unwrap(), (3, &8));
        assert_eq!(it.next().unwrap(), (6, &16));
        assert_eq!(it.next().unwrap(), (10, &21));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_rev_iterator() {
        let mut m = VecMap::<usize, _>::new();

        assert!(m.insert(0, 1).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        let mut it = m.iter().rev();
        assert_eq!(it.next().unwrap(), (10, &11));
        assert_eq!(it.next().unwrap(), (6, &10));
        assert_eq!(it.next().unwrap(), (3, &5));
        assert_eq!(it.next().unwrap(), (1, &2));
        assert_eq!(it.next().unwrap(), (0, &1));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_mut_rev_iterator() {
        let mut m = VecMap::<usize, _>::new();

        assert!(m.insert(0, 1).is_none());
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(3, 5).is_none());
        assert!(m.insert(6, 10).is_none());
        assert!(m.insert(10, 11).is_none());

        for (k, v) in m.iter_mut().rev() {
            *v += k as isize;
        }

        let mut it = m.iter();
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.next().unwrap(), (1, &3));
        assert_eq!(it.next().unwrap(), (3, &8));
        assert_eq!(it.next().unwrap(), (6, &16));
        assert_eq!(it.next().unwrap(), (10, &21));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_move_iter() {
        let mut m: VecMap<usize, Box<_>> = VecMap::<usize, _>::new();
        m.insert(1, Box::new(2));
        let mut called = false;
        for (k, v) in m {
            assert!(!called);
            called = true;
            assert_eq!(k, 1);
            assert_eq!(v, Box::new(2));
        }
        assert!(called);
    }

    #[test]
    fn test_drain_iterator() {
        let mut map = VecMap::<usize, _>::new();
        map.insert(1, "a");
        map.insert(3, "c");
        map.insert(2, "b");

        let vec: Vec<_> = map.drain().collect();

        assert_eq!(vec, [(1, "a"), (2, "b"), (3, "c")]);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_append() {
        let mut a = VecMap::<usize, _>::new();
        a.insert(1, "a");
        a.insert(2, "b");
        a.insert(3, "c");

        let mut b = VecMap::<usize, _>::new();
        b.insert(3, "d"); // Overwrite element from a
        b.insert(4, "e");
        b.insert(5, "f");

        a.append(&mut b);

        assert_eq!(a.len(), 5);
        assert_eq!(b.len(), 0);
        // Capacity shouldn't change for possible reuse
        assert!(b.capacity() >= 4);

        assert_eq!(a[1], "a");
        assert_eq!(a[2], "b");
        assert_eq!(a[3], "d");
        assert_eq!(a[4], "e");
        assert_eq!(a[5], "f");
    }

    #[test]
    fn test_split_off() {
        // Split within the key range
        let mut a = VecMap::<usize, _>::new();
        a.insert(1, "a");
        a.insert(2, "b");
        a.insert(3, "c");
        a.insert(4, "d");

        let b = a.split_off(3);

        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);

        assert_eq!(a[1], "a");
        assert_eq!(a[2], "b");

        assert_eq!(b[3], "c");
        assert_eq!(b[4], "d");

        // Split at 0
        a.clear();
        a.insert(1, "a");
        a.insert(2, "b");
        a.insert(3, "c");
        a.insert(4, "d");

        let b = a.split_off(0);

        assert_eq!(a.len(), 0);
        assert_eq!(b.len(), 4);
        assert_eq!(b[1], "a");
        assert_eq!(b[2], "b");
        assert_eq!(b[3], "c");
        assert_eq!(b[4], "d");

        // Split behind max_key
        a.clear();
        a.insert(1, "a");
        a.insert(2, "b");
        a.insert(3, "c");
        a.insert(4, "d");

        let b = a.split_off(5);

        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 0);
        assert_eq!(a[1], "a");
        assert_eq!(a[2], "b");
        assert_eq!(a[3], "c");
        assert_eq!(a[4], "d");
    }

    #[test]
    fn test_show() {
        let mut map = VecMap::<usize, _>::new();
        let empty = VecMap::<usize, i32>::new();

        map.insert(1, 2);
        map.insert(3, 4);

        let map_str = format!("{:?}", map);
        assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_clone() {
        let mut a = VecMap::<usize, _>::new();

        a.insert(1, 'x');
        a.insert(4, 'y');
        a.insert(6, 'z');

        assert_eq!(
            a.clone().iter().collect::<Vec<_>>(),
            [(1, &'x'), (4, &'y'), (6, &'z')]
        );
    }

    #[test]
    fn test_eq() {
        let mut a = VecMap::<usize, _>::new();
        let mut b = VecMap::<usize, _>::new();

        assert!(a == b);
        assert!(a.insert(0, 5).is_none());
        assert!(a != b);
        assert!(b.insert(0, 4).is_none());
        assert!(a != b);
        assert!(a.insert(5, 19).is_none());
        assert!(a != b);
        assert!(!b.insert(0, 5).is_none());
        assert!(a != b);
        assert!(b.insert(5, 19).is_none());
        assert!(a == b);

        a = VecMap::<usize, _>::new();
        b = VecMap::with_capacity(1);
        assert!(a == b);
    }

    #[test]
    fn test_lt() {
        let mut a = VecMap::<usize, _>::new();
        let mut b = VecMap::<usize, _>::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(2, 5).is_none());
        assert!(a < b);
        assert!(a.insert(2, 7).is_none());
        assert!(!(a < b) && b < a);
        assert!(b.insert(1, 0).is_none());
        assert!(b < a);
        assert!(a.insert(0, 6).is_none());
        assert!(a < b);
        assert!(a.insert(6, 2).is_none());
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = VecMap::<usize, _>::new();
        let mut b = VecMap::<usize, _>::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1, 1).is_none());
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2).is_none());
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_hash() {
        fn hash<T: Hash>(t: &T) -> u64 {
            let mut s = SipHasher::new();
            t.hash(&mut s);
            s.finish()
        }

        let mut x = VecMap::<usize, _>::new();
        let mut y = VecMap::<usize, _>::new();

        assert!(hash(&x) == hash(&y));
        x.insert(1, 'a');
        x.insert(2, 'b');
        x.insert(3, 'c');

        y.insert(3, 'c');
        y.insert(2, 'b');
        y.insert(1, 'a');

        assert!(hash(&x) == hash(&y));

        x.insert(1000, 'd');
        x.remove(1000);

        assert!(hash(&x) == hash(&y));
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')];

        let map: VecMap<usize, _> = xs.iter().cloned().collect();

        for &(k, v) in &xs {
            assert_eq!(map.get(k), Some(&v));
        }
    }

    #[test]
    fn test_index() {
        let mut map = VecMap::<usize, _>::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[3], 4);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = VecMap::<usize, _>::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[4];
    }

    #[test]
    fn test_entry() {
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: VecMap<usize, _> = xs.iter().cloned().collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }

        assert_eq!(map.get(1).unwrap(), &100);
        assert_eq!(map.len(), 6);

        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                *v *= 10;
            }
        }

        assert_eq!(map.get(2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.remove(), 30);
            }
        }

        assert_eq!(map.get(3), None);
        assert_eq!(map.len(), 5);

        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.insert(1000), 1000);
            }
        }

        assert_eq!(map.get(10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_extend_ref() {
        let mut a = VecMap::<usize, _>::new();
        a.insert(1, "one");
        let mut b = VecMap::<usize, _>::new();
        b.insert(2, "two");
        b.insert(3, "three");

        a.extend(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a[&1], "one");
        assert_eq!(a[&2], "two");
        assert_eq!(a[&3], "three");
    }

    #[test]
    #[cfg(feature = "eders")]
    fn test_serde() {
        use serde::{Deserialize, Serialize};
        fn impls_serde_traits<'de, S: Serialize + Deserialize<'de>>() {}

        impls_serde_traits::<VecMap<u32>>();
    }

    #[test]
    fn test_retain() {
        let mut map = VecMap::<usize, _>::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        map.retain(|k, v| match k {
            1 => false,
            2 => {
                *v = "two changed";
                true
            }
            3 => false,
            _ => panic!(),
        });

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(1), None);
        assert_eq!(map[2], "two changed");
        assert_eq!(map.get(3), None);
    }
}
