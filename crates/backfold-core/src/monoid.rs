//! Core associative monoid abstraction for O(log T) retractable streaming inference.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An associative algebraic summary over a chronologically ordered interval of events.
///
/// # Mathematical Contract
/// For every context `ctx` and summaries `a, b, c`:
/// 1. **Associativity:** `combine(combine(a, b, ctx), c, ctx) == combine(a, combine(b, c, ctx), ctx)`
///    (up to floating-point rounding tolerance for real-valued operators).
/// 2. **Identity:** `combine(identity(ctx), a, ctx) == a == combine(a, identity(ctx), ctx)`.
/// 3. **Order-sensitivity:** `combine` is **not** assumed commutative; `left` always precedes `right`
///    in chronological [`crate::EventKey`] order.
pub trait Monoid: Clone + Send + Sync + 'static {
    /// Immutable operator context (e.g., `()` for context-free monoids, decay rate `f64` for EWMA,
    /// or `(dim, decay)` for dynamic-dimension Ridge). Stored once on the tree root (zero per-node overhead).
    type Ctx: Clone + core::fmt::Debug + Send + Sync + 'static;

    /// Empty-interval identity element.
    fn identity(ctx: &Self::Ctx) -> Self;

    /// Associatively combine an earlier interval `left` with a later interval `right`.
    fn combine(left: &Self, right: &Self, ctx: &Self::Ctx) -> Self;

    /// In-place associative accumulation `self = combine(self, right, ctx)`.
    #[inline]
    fn combine_assign(&mut self, right: &Self, ctx: &Self::Ctx) {
        *self = Self::combine(self, right, ctx);
    }

    /// Fast check for whether this summary represents zero events.
    fn is_identity(&self) -> bool;
}

/// Product monoid combining two independent summaries in a single `FoldTree` pass.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PairMonoid<A, B> {
    pub first: A,
    pub second: B,
}

impl<A: Monoid, B: Monoid> Monoid for PairMonoid<A, B> {
    type Ctx = (A::Ctx, B::Ctx);

    #[inline]
    fn identity(ctx: &Self::Ctx) -> Self {
        Self {
            first: A::identity(&ctx.0),
            second: B::identity(&ctx.1),
        }
    }

    #[inline]
    fn combine(left: &Self, right: &Self, ctx: &Self::Ctx) -> Self {
        if left.is_identity() {
            return right.clone();
        }
        if right.is_identity() {
            return left.clone();
        }
        Self {
            first: A::combine(&left.first, &right.first, &ctx.0),
            second: B::combine(&left.second, &right.second, &ctx.1),
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.first.is_identity() && self.second.is_identity()
    }
}

/// Input validation error for non-finite values, invalid dimensions, or invalid parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum BackfoldError {
    NonFiniteInput(&'static str),
    InvalidParameter(&'static str),
    DimensionMismatch { expected: usize, got: usize },
    SingularMatrix(&'static str),
}

impl core::fmt::Display for BackfoldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteInput(msg) => write!(f, "non-finite input: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::SingularMatrix(msg) => write!(f, "singular or non-PD matrix: {msg}"),
        }
    }
}

impl std::error::Error for BackfoldError {}

#[inline]
pub fn check_finite(v: f64, label: &'static str) -> Result<f64, BackfoldError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(BackfoldError::NonFiniteInput(label))
    }
}

#[inline]
pub fn check_all_finite(xs: &[f64], label: &'static str) -> Result<(), BackfoldError> {
    for &x in xs {
        if !x.is_finite() {
            return Err(BackfoldError::NonFiniteInput(label));
        }
    }
    Ok(())
}
