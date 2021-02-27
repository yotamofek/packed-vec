#![allow(incomplete_features)]
#![feature(is_sorted, const_generics, const_evaluatable_checked)]

use std::{
    convert::{TryFrom, TryInto},
    fmt::{Debug, Display},
    marker::PhantomData,
    mem::size_of,
};

use num_traits::{NumAssign, PrimInt, Unsigned};

pub trait NeBytes<const SIZE: usize> {
    fn from_ne_bytes(bytes: [u8; SIZE]) -> Self;
    fn into_ne_bytes(self) -> [u8; SIZE];
}

pub trait Num:
    PrimInt + Unsigned + NumAssign + Debug + Display + NeBytes<{ size_of::<Self>() }>
where
    [u8; size_of::<Self>()]: ,
{
}

impl<N> Num for N
where
    N: PrimInt + Unsigned + NumAssign + Debug + Display + NeBytes<{ size_of::<Self>() }>,
    [u8; size_of::<Self>()]: ,
{
}

macro_rules! impl_ne_bytes {
    ($t:ident) => {
        impl NeBytes<{ size_of::<Self>() }> for $t {
            fn from_ne_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
                Self::from_ne_bytes(bytes)
            }

            fn into_ne_bytes(self) -> [u8; size_of::<Self>()] {
                Self::to_ne_bytes(self)
            }
        }
    };
    ($t: ident $($ts:ident)+) => {
        impl_ne_bytes!($t);
        impl_ne_bytes!($($ts)+);
    };
}

impl_ne_bytes!(u64 u32 u16 u8);

#[derive(Debug)]
pub struct PackedVec<V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    inner: Vec<u8>,

    _v_marker: PhantomData<V>,
    _d_marker: PhantomData<D>,
}

impl<V, D> Default for PackedVec<V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    fn default() -> Self {
        Self::new()
    }
}

enum Delta<V, D> {
    Small(D),
    Big(V),
}

impl<V, D> Delta<V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    fn compress(delta: V) -> Self {
        if let Ok(small_delta) = D::try_from(delta) {
            Delta::Small(small_delta)
        } else {
            Delta::Big(delta)
        }
    }

    fn size_of_val(&self) -> usize {
        match self {
            Delta::Small(..) => size_of::<D>(),
            Delta::Big(..) => size_of::<D>() + size_of::<V>(),
        }
    }

    fn as_value(&self) -> V {
        match *self {
            Delta::Small(small) => <V as From<D>>::from(small),
            Delta::Big(value) => value,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        match *self {
            Delta::Small(small) => small.into_ne_bytes().to_vec(),
            Delta::Big(big) => {
                let mut bytes = D::into_ne_bytes(D::min_value()).to_vec();
                bytes.extend_from_slice(&V::into_ne_bytes(big));
                bytes
            }
        }
    }
}

fn read_delta<V, D>(data: &[u8]) -> Delta<V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    let delta = D::from_ne_bytes(data[0..size_of::<D>()].try_into().unwrap());

    if !delta.is_zero() {
        Delta::Small(delta)
    } else {
        let delta = V::from_ne_bytes(
            data[size_of::<D>()..size_of::<D>() + size_of::<V>()]
                .try_into()
                .unwrap(),
        );

        Delta::Big(delta)
    }
}

impl<V, D> PackedVec<V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            _v_marker: Default::default(),
            _d_marker: Default::default(),
        }
    }

    pub fn iter(&self) -> PackedVecIter<'_, V, D> {
        PackedVecIter {
            raw: self.inner.as_slice(),
            cur_val: V::min_value(),

            _marker: Default::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    #[inline]
    pub fn push(&mut self, el: V) -> InsertPosition {
        let mut extender = PackedVecExtender {
            vec: self,
            cur_val: V::min_value(),
            pos: 0,
            idx: 0,

            _v_marker: Default::default(),
            _d_marker: Default::default(),
        };

        unsafe { extender.push(el) }
    }

    #[inline]
    pub fn extender(&mut self) -> PackedVecExtender<'_, V, D> {
        PackedVecExtender {
            vec: self,
            cur_val: V::min_value(),
            pos: 0,
            idx: 0,

            _v_marker: Default::default(),
            _d_marker: Default::default(),
        }
    }

    /// # Safety
    /// Passing a slice that is either not completely sorted or not completely deduped
    /// can corrupt this structure's internal storage
    pub unsafe fn extend_from_sorted_unique_slice(&mut self, elements: &[V]) {
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

pub struct PackedVecExtender<'a, V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    vec: &'a mut PackedVec<V, D>,
    cur_val: V,
    pos: usize,
    idx: usize,

    _v_marker: PhantomData<V>,
    _d_marker: PhantomData<D>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum InsertPosition {
    Position(usize),
    AlreadyExists(usize),
}

impl InsertPosition {
    pub fn into_position(self) -> usize {
        match self {
            InsertPosition::Position(pos) | InsertPosition::AlreadyExists(pos) => pos,
        }
    }
}

impl<'a, V, D> PackedVecExtender<'a, V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    unsafe fn read_delta(&mut self) -> Option<Delta<V, D>> {
        let Self {
            vec: PackedVec { ref inner, .. },
            pos,
            ..
        } = self;

        let slice = &inner[*pos..];

        if slice.is_empty() {
            return None;
        }

        let delta = read_delta(slice);

        Some(delta)
    }

    unsafe fn advance(&mut self, delta: Delta<V, D>) {
        self.idx += 1;
        self.pos += delta.size_of_val();
        self.cur_val += delta.as_value();
    }

    /// # Safety
    /// Using this function on a set of elements that is either not completely sorted or not completely
    /// deduped can corrupt this structure's internal storage
    pub unsafe fn push(&mut self, item: V) -> InsertPosition {
        while let Some(delta) = self.read_delta() {
            let new_val = self.cur_val + delta.as_value();

            if item < new_val {
                break;
            }

            if item == new_val {
                return InsertPosition::AlreadyExists(self.idx);
            }

            self.advance(delta);
        }

        let delta = {
            let Self {
                cur_val,
                idx,
                vec: PackedVec { inner, .. },
                pos,
                ..
            } = self;

            let delta = Delta::<V, D>::compress(item - *cur_val);

            let _ = inner.splice(*pos..*pos, delta.to_bytes());

            *cur_val = item;
            *pos += delta.size_of_val();
            *idx += 1;

            delta
        };

        if let Some(next_delta) = self.read_delta() {
            let Self {
                pos,
                vec: PackedVec { inner, .. },
                ..
            } = *self;

            let new_delta = Delta::<V, D>::compress(next_delta.as_value() - delta.as_value());

            let _ = inner.splice(pos..pos + next_delta.size_of_val(), new_delta.to_bytes());
        }

        InsertPosition::Position(self.idx - 1)
    }
}

impl<V, D> Extend<V> for PackedVec<V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    fn extend<T: IntoIterator<Item = V>>(&mut self, iter: T) {
        iter.into_iter().for_each(|el| {
            self.push(el);
        });
    }
}

pub struct PackedVecIter<'s, V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    raw: &'s [u8],
    cur_val: V,

    _marker: PhantomData<D>,
}

impl<'s, V, D> Iterator for PackedVecIter<'s, V, D>
where
    V: Num + From<D>,
    D: Num + TryFrom<V>,
    [u8; size_of::<V>()]: ,
    [u8; size_of::<D>()]: ,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        if self.raw.is_empty() {
            None
        } else {
            let delta = read_delta::<V, D>(&self.raw);
            self.raw = &self.raw[delta.size_of_val()..];

            let delta = delta.as_value();

            self.cur_val += delta;
            Some(self.cur_val)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, iter};

    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn it_works() {
        let mut vec = PackedVec::<u64, u8>::new();

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
        let mut vec = PackedVec::<u64, u8>::new();

        #[cfg(debug_assertions)]
        unsafe {
            vec.extend_from_sorted_unique_slice(&[23, 54, 54, 65]);
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn test_debug_assertion_sorted() {
        let mut vec = PackedVec::<u64, u8>::new();

        #[cfg(debug_assertions)]
        unsafe {
            vec.extend_from_sorted_unique_slice(&[1, 0]);
        }
    }

    #[test]
    fn test_ratio() {
        let mut vec = PackedVec::<u32, u8>::new();

        let mut rng = thread_rng();

        let mut input = iter::from_fn(move || Some(rng.gen()))
            .take(2_000_000)
            .collect::<Vec<_>>();

        input.sort_unstable();
        input.dedup();

        unsafe {
            vec.extend_from_sorted_unique_slice(&input);
        }

        println!("Original length: {}", input.len() * size_of::<u64>());
        println!("u32 delta length: {}", vec.inner.len());
    }
}
