#![feature(is_sorted)]

use std::{
    fmt::{self, Debug},
    io,
};

use prefix_varint::{read_varint, write_varint};

pub struct PackedVec {
    inner: Vec<u8>,
}

impl Debug for PackedVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vec = self.iter().collect::<Vec<_>>();

        vec.fmt(f)
    }
}

impl Clone for PackedVec {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Default for PackedVec {
    fn default() -> Self {
        Self::new()
    }
}

impl PackedVec {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn with_capacity(cap_in_bytes: usize) -> Self {
        Self {
            inner: Vec::with_capacity(cap_in_bytes),
        }
    }

    pub fn from_bytes(inner: Vec<u8>) -> Self {
        Self { inner }
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }

    #[inline]
    pub fn iter(&self) -> PackedVecIter<'_> {
        PackedVecIter {
            raw: io::Cursor::new(self.inner.as_slice()),
            cur_val: 0,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    #[inline]
    pub fn push(&mut self, el: u64) -> InsertPosition {
        let mut extender = self.extender();

        unsafe { extender.push(el) }
    }

    #[inline]
    pub fn extender(&mut self) -> PackedVecExtender<'_> {
        PackedVecExtender {
            vec: self,
            cur_val: 0,
            pos: 0,
            idx: 0,
        }
    }

    /// # Safety
    /// Passing a slice that is either not completely sorted or not completely deduped
    /// can corrupt this structure's internal storage
    #[inline]
    pub unsafe fn extend_from_sorted_unique_slice(&mut self, elements: &[u64]) {
        debug_assert!(elements.is_sorted());
        debug_assert!({
            let cur = elements.iter().skip(1);
            let prev = elements.iter();

            cur.zip(prev).all(|(cur, prev)| cur != prev)
        });

        let mut extender = self.extender();

        elements.iter().copied().for_each(|el| {
            extender.push(el);
        })
    }
}

pub struct PackedVecExtender<'a> {
    vec: &'a mut PackedVec,
    cur_val: u64,
    pos: usize,
    idx: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum InsertPosition {
    Position(usize),
    AlreadyExists(usize),
}

impl InsertPosition {
    #[inline]
    pub fn into_position(self) -> usize {
        match self {
            InsertPosition::Position(pos) | InsertPosition::AlreadyExists(pos) => pos,
        }
    }
}

fn write_varint_to_buf(n: u64) -> io::Result<Vec<u8>> {
    let mut buf = vec![];

    write_varint(n, &mut buf)?;

    Ok(buf)
}

impl<'a> PackedVecExtender<'a> {
    unsafe fn read_delta(&mut self) -> Option<(u64, usize)> {
        let Self {
            vec: PackedVec { ref inner, .. },
            pos,
            ..
        } = self;

        let slice = &inner[*pos..];

        if slice.is_empty() {
            return None;
        }

        let mut cursor = io::Cursor::new(slice);

        let delta = read_varint(&mut cursor).unwrap_unchecked();

        Some((delta, cursor.position() as usize))
    }

    /// # Safety
    /// Using this function on a set of elements that is either not completely sorted or not completely
    /// deduped can corrupt this structure's internal storage
    #[inline]
    pub unsafe fn push(&mut self, item: u64) -> InsertPosition {
        while let Some((delta, size)) = self.read_delta() {
            let new_val = self.cur_val + delta;

            if item < new_val {
                break;
            }

            if item == new_val {
                return InsertPosition::AlreadyExists(self.idx);
            }

            self.idx += 1;
            self.pos += size;
            self.cur_val += delta;
        }

        let delta = {
            let Self {
                cur_val,
                idx,
                vec: PackedVec { inner, .. },
                pos,
                ..
            } = self;

            let delta = item - *cur_val;

            let encoded_delta = write_varint_to_buf(delta).unwrap();
            let encoded_len = encoded_delta.len();

            let _ = inner.splice(*pos..*pos, encoded_delta);

            *cur_val = item;
            *pos += encoded_len;
            *idx += 1;

            delta
        };

        if let Some((next_delta, size)) = self.read_delta() {
            let Self {
                pos,
                vec: PackedVec { inner, .. },
                ..
            } = *self;

            let new_delta = next_delta - delta;

            let _ = inner.splice(pos..pos + size, write_varint_to_buf(new_delta).unwrap());
        }

        InsertPosition::Position(self.idx - 1)
    }
}

impl Extend<u64> for PackedVec {
    #[inline]
    fn extend<T: IntoIterator<Item = u64>>(&mut self, iter: T) {
        iter.into_iter().for_each(|el| {
            self.push(el);
        });
    }
}

pub struct PackedVecIter<'s> {
    raw: io::Cursor<&'s [u8]>,
    cur_val: u64,
}

impl<'s> Iterator for PackedVecIter<'s> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.raw.position() == self.raw.get_ref().len() as u64 {
            None
        } else {
            self.cur_val += unsafe { read_varint(&mut self.raw).unwrap_unchecked() };

            Some(self.cur_val)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, iter, mem::size_of};

    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn it_works() {
        let mut vec = PackedVec::new();

        let expected = [6345, 234, 34632, 34633, 656, 57];

        {
            assert!(vec.is_empty());
            vec.extend(expected.iter().cloned());
            assert!(!vec.is_empty());
            assert_eq!(vec.len(), expected.len());
            assert_eq!(
                vec.iter().collect::<HashSet<_>>(),
                expected.iter().copied().collect()
            );
        }

        let expected2 = [34, 563, 53534, 854593];

        {
            vec.extend(expected2.iter().cloned());
            assert_eq!(
                vec.len(),
                expected
                    .iter()
                    .chain(expected2.iter())
                    .copied()
                    .collect::<HashSet<_>>()
                    .len()
            );
            assert_eq!(
                vec.iter().collect::<HashSet<_>>(),
                expected.iter().chain(expected2.iter()).copied().collect()
            );
        }

        let len = vec.len();
        vec.extend([563, 34].iter().cloned());
        assert_eq!(len, vec.len());

        assert_eq!(
            vec.iter().collect::<HashSet<_>>(),
            expected.iter().chain(expected2.iter()).copied().collect()
        );

        let len = vec.len();
        let mut extender = vec.extender();

        unsafe {
            assert_eq!(extender.push(30), InsertPosition::Position(0));
            assert_eq!(extender.push(34), InsertPosition::AlreadyExists(1));
            assert_eq!(extender.push(235), InsertPosition::Position(4));
            assert_eq!(extender.push(854594), InsertPosition::Position(len + 2));
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn test_debug_assertion_deduped() {
        let mut vec = PackedVec::new();

        #[cfg(debug_assertions)]
        unsafe {
            vec.extend_from_sorted_unique_slice(&[23, 54, 54, 65]);
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn test_debug_assertion_sorted() {
        let mut vec = PackedVec::new();

        #[cfg(debug_assertions)]
        unsafe {
            vec.extend_from_sorted_unique_slice(&[1, 0]);
        }
    }

    #[test]
    fn test_ratio() {
        let mut vec = PackedVec::new();

        let mut rng = thread_rng();

        let mut input = iter::from_fn(move || Some(rng.gen()))
            .take(250_000)
            .collect::<Vec<_>>();

        input.sort_unstable();
        input.dedup();

        unsafe {
            vec.extend_from_sorted_unique_slice(&input);
        }

        println!("Original len: {}", input.len() * size_of::<u64>());
        println!("Packed len:   {}", vec.inner.len());
    }
}
