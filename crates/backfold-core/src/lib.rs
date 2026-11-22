//! `backfold-core`: Persistent O(log T) associative-monoid engine for out-of-order,
//! retractable, forkable online statistical inference and agent stream compression.

#![forbid(unsafe_code)]

pub mod digest;
pub mod key;
pub mod monoid;
pub mod ops;
pub mod swarm;
pub mod tree;

pub use key::EventKey;
pub use monoid::{BackfoldError, Monoid, PairMonoid};
pub use tree::FoldTree;
