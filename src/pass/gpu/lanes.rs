use alloc::vec;
use alloc::vec::Vec;

use mlirformat::Type;
use mlirformat::spirv_types::SpirvType;

use crate::{ConversionRewriter, OpId, ValueId, Xe3Types};

/// Lowering a vector op lane by lane: its operands' lanes in, the result's lanes out
pub trait Lanes {
    fn lane_positions(&self, op: OpId, name: &str) -> Option<Vec<usize>>;
    fn lane_position(&self, op: OpId) -> Option<usize>;
    fn lanes_each(&mut self, op: OpId, o: &[Vec<ValueId>], f: impl FnMut(&mut Self, &[ValueId]) -> ValueId) -> bool;
    fn lanes_extract(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool;
    fn lanes_insert(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool;
    fn lanes_from_elements(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool;
    fn lanes_broadcast(&mut self, op: OpId, o: &[Vec<ValueId>], lanes: fn(&Type<'_>) -> Option<usize>) -> bool;
    fn lanes_shuffle(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool;
}

impl Lanes for ConversionRewriter<'_, '_, '_> {
    fn lane_positions(&self, op: OpId, name: &str) -> Option<Vec<usize>> {
        self.rw.body[op].get(name)?.dense_values().map(|v| v.into_iter().map(|i| i as usize).collect())
    }

    fn lane_position(&self, op: OpId) -> Option<usize> {
        match self.lane_positions(op, "static_position")?[..] {
            [at] => Some(at),
            _ => None,
        }
    }

    fn lanes_each(&mut self, op: OpId, o: &[Vec<ValueId>], mut f: impl FnMut(&mut Self, &[ValueId]) -> ValueId) -> bool {
        let n = o[0].len();
        if o.iter().any(|v| v.len() != n) {
            return false;
        }
        let values: Vec<ValueId> = (0..n).map(|i| f(self, &o.iter().map(|v| v[i]).collect::<Vec<_>>())).collect();
        self.replace(op, vec![values]);
        true
    }

    fn lanes_extract(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(v) = self.lane_position(op).and_then(|at| o[0].get(at)).copied() else { return false };
        self.replace(op, vec![vec![v]]);
        true
    }

    fn lanes_insert(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(at) = self.lane_position(op) else { return false };
        let ([v], Some(dest)) = (&o[0][..], o.get(1)) else { return false };
        if at >= dest.len() {
            return false;
        }
        let mut dest = dest.clone();
        dest[at] = *v;
        self.replace(op, vec![dest]);
        true
    }

    fn lanes_from_elements(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool {
        self.replace(op, vec![o.concat()]);
        true
    }

    fn lanes_broadcast(&mut self, op: OpId, o: &[Vec<ValueId>], lanes: fn(&Type<'_>) -> Option<usize>) -> bool {
        let r = self.rw.body[op].results[0];
        let (Some(n), [v]) = (lanes(&self.rw.body[r].ty), &o[0][..]) else { return false };
        self.replace(op, vec![vec![*v; n]]);
        true
    }

    fn lanes_shuffle(&mut self, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(mask) = self.lane_positions(op, "mask") else { return false };
        let all = o.concat();
        let Some(values) = mask.iter().map(|i| all.get(*i).copied()).collect::<Option<Vec<_>>>() else { return false };
        self.replace(op, vec![values]);
        true
    }
}

/// SPIR-V's matrix arithmetic lane by lane: a matrix is its columns' lanes one after another (column-major), a vector
/// one column
pub struct Matrices;

impl Matrices {
    /// A value of type `t`'s columns and rows
    fn shape(t: &Type<'_>) -> Option<(usize, usize)> {
        match SpirvType::try_from(t) {
            Ok(SpirvType::Matrix { cols, column }) => Some((cols as usize, Xe3Types::lanes(&column)?)),
            _ => Some((1, Xe3Types::lanes(t)?)),
        }
    }

    /// For `name`, one of SPIR-V's matrix and vector products of operands of types `a` and `b`, each result lane's
    /// terms: the lanes of `a` and of `b` whose products it sums
    pub fn products(name: &str, a: &Type<'_>, b: &Type<'_>) -> Option<Vec<Vec<(usize, usize)>>> {
        let ((ac, ar), (bc, br)) = (Self::shape(a)?, Self::shape(b)?);
        Some(match name {
            "spirv.MatrixTimesVector" if bc == 1 && br == ac => (0..ar).map(|r| (0..ac).map(|j| (j * ar + r, j)).collect()).collect(),
            "spirv.VectorTimesMatrix" if ac == 1 && ar == br => (0..bc).map(|c| (0..ar).map(|i| (i, c * br + i)).collect()).collect(),
            "spirv.MatrixTimesMatrix" if br == ac => (0..bc).flat_map(|c| (0..ar).map(move |r| (0..ac).map(|j| (j * ar + r, c * br + j)).collect())).collect(),
            "spirv.OuterProduct" if ac == 1 && bc == 1 => (0..br).flat_map(|j| (0..ar).map(move |i| vec![(i, j)])).collect(),
            "spirv.MatrixTimesScalar" | "spirv.VectorTimesScalar" if (bc, br) == (1, 1) => (0..ac * ar).map(|i| vec![(i, 0)]).collect(),
            _ => return None,
        })
    }

    /// `spirv.Transpose` of a matrix of type `t`: each result lane's lane of it
    pub fn transpose(t: &Type<'_>) -> Option<Vec<usize>> {
        let (cols, rows) = Self::shape(t)?;
        Some((0..rows).flat_map(|r| (0..cols).map(move |c| c * rows + r)).collect())
    }
}
