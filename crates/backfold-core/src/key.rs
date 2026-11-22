//! Causal event keys for total chronological ordering and echo-proof deduplication.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Total chronological key for an event in a `FoldTree`.
///
/// Ordering is lexicographic: `(event_time, source_id, seq)`.
/// Because every event from a producer carries `(event_time, source_id, seq)`,
/// re-inserting or gossiping the same observation across multiple agents is
/// strictly **idempotent** (never double-counted).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EventKey {
    /// Event timestamp (e.g., Unix nanoseconds, milliseconds, or discrete tick index).
    pub event_time: i64,
    /// Producer / sensor / agent identifier (0 for single-source streams).
    pub source_id: u64,
    /// Per-source sequence or tie-breaker within the same `event_time`.
    pub seq: u64,
}

impl EventKey {
    /// Minimum possible key (lower sentinel for range queries).
    pub const MIN: Self = Self {
        event_time: i64::MIN,
        source_id: 0,
        seq: 0,
    };

    /// Maximum possible key (upper sentinel for range queries).
    pub const MAX: Self = Self {
        event_time: i64::MAX,
        source_id: u64::MAX,
        seq: u64::MAX,
    };

    /// Construct a single-source key at `event_time`.
    #[inline]
    pub const fn at(event_time: i64) -> Self {
        Self {
            event_time,
            source_id: 0,
            seq: 0,
        }
    }

    /// Construct a multi-source echo-proof key at `(event_time, source_id, seq)`.
    #[inline]
    pub const fn sourced(event_time: i64, source_id: u64, seq: u64) -> Self {
        Self {
            event_time,
            source_id,
            seq,
        }
    }

    /// Earliest key with timestamp `event_time` (inclusive lower bound for time ranges).
    #[inline]
    pub const fn time_start(event_time: i64) -> Self {
        Self {
            event_time,
            source_id: 0,
            seq: 0,
        }
    }

    /// Latest key with timestamp `event_time` (inclusive upper bound for time ranges).
    #[inline]
    pub const fn time_end(event_time: i64) -> Self {
        Self {
            event_time,
            source_id: u64::MAX,
            seq: u64::MAX,
        }
    }
}
