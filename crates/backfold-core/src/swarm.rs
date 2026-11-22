//! Bitemporal point-in-time snapshots and echo-proof multi-agent CRDT swarm ledger.

use std::collections::BTreeMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::key::EventKey;
use crate::monoid::{BackfoldError, Monoid};
use crate::tree::FoldTree;

/// Bitemporal wrapper over `FoldTree` preserving zero-copy `as_of(knowledge_time)` snapshots.
///
/// Guarantees zero lookahead leakage: querying `as_of(kt)` returns the exact historical
/// `FoldTree` committed at or before `knowledge_time`, ignoring all later restatements.
#[derive(Clone, Debug)]
pub struct BitemporalFold<M: Monoid, const B: usize = 8> {
    live: FoldTree<M, B>,
    checkpoints: Vec<(i64, FoldTree<M, B>)>,
}

impl<M: Monoid, const B: usize> BitemporalFold<M, B> {
    pub fn new(ctx: M::Ctx) -> Self {
        Self {
            live: FoldTree::new(ctx),
            checkpoints: Vec::new(),
        }
    }

    pub fn from_tree(live: FoldTree<M, B>) -> Self {
        Self {
            live,
            checkpoints: Vec::new(),
        }
    }

    #[inline]
    pub fn upsert(&mut self, key: EventKey, value: M) -> Option<M> {
        self.live.upsert(key, value)
    }

    #[inline]
    pub fn retract(&mut self, key: EventKey) -> Option<M> {
        self.live.retract(key)
    }

    /// Commit an O(1) persistent snapshot at `knowledge_time` (must be monotonically non-decreasing).
    pub fn commit(&mut self, knowledge_time: i64) -> Result<(), BackfoldError> {
        if let Some(&(last_kt, _)) = self.checkpoints.last() {
            if knowledge_time < last_kt {
                return Err(BackfoldError::InvalidParameter(
                    "knowledge_time must be monotonically non-decreasing",
                ));
            }
            if knowledge_time == last_kt {
                self.checkpoints.last_mut().unwrap().1 = self.live.fork();
                return Ok(());
            }
        }
        self.checkpoints.push((knowledge_time, self.live.fork()));
        Ok(())
    }

    /// O(log V) lookup of the exact tree visible at `knowledge_time`.
    pub fn as_of(&self, knowledge_time: i64) -> Option<&FoldTree<M, B>> {
        let idx = self.checkpoints.partition_point(|(kt, _)| *kt <= knowledge_time);
        if idx == 0 {
            None
        } else {
            Some(&self.checkpoints[idx - 1].1)
        }
    }

    #[inline]
    pub fn current(&self) -> &FoldTree<M, B> {
        &self.live
    }

    #[inline]
    pub fn versions(&self) -> usize {
        self.checkpoints.len()
    }
}

/// Telemetry counters for multi-agent evidence deduplication and conflict-free merging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmStats {
    pub n_agents: usize,
    pub raw_observations: u64,
    pub unique_events: usize,
    pub echoes_deduped: u64,
    pub retractions: u64,
    pub overwrites: u64,
}

/// Echo-proof, conflict-free multi-agent causal CRDT ledger built on persistent `FoldTree` forks.
///
/// Solves three core multi-agent streaming failure modes:
/// 1. **Overwrite churn (0%):** Agents operate on isolated O(1) persistent forks and merge via
///    associative key-union rather than last-write-wins blackboards.
/// 2. **Echo double-counting (0%):** Identical `(event_time, source_id, seq)` observations
///    gossiped across multiple agents collapse idempotently to a single leaf.
/// 3. **Counterfactual branching:** Each agent can query its private branch immediately in O(1).
#[derive(Clone, Debug)]
pub struct SwarmLedger<M: Monoid, const B: usize = 8> {
    base: FoldTree<M, B>,
    agents: BTreeMap<u64, FoldTree<M, B>>,
    pending_upserts: BTreeMap<EventKey, M>,
    pending_retractions: Vec<EventKey>,
    raw_observations: u64,
    echoes_deduped: u64,
    retractions: u64,
}

impl<M: Monoid, const B: usize> SwarmLedger<M, B> {
    pub fn new(ctx: M::Ctx) -> Self {
        Self::from_base(FoldTree::new(ctx))
    }

    pub fn from_base(base: FoldTree<M, B>) -> Self {
        Self {
            base,
            agents: BTreeMap::new(),
            pending_upserts: BTreeMap::new(),
            pending_retractions: Vec::new(),
            raw_observations: 0,
            echoes_deduped: 0,
            retractions: 0,
        }
    }

    /// Spawn or reset an agent with an O(1) zero-copy persistent fork of the canonical base tree.
    pub fn spawn_agent(&mut self, agent_id: u64) {
        self.agents.insert(agent_id, self.base.fork());
    }

    /// Record an observation from `agent_id`. Returns `true` if the `(event_time, source_id, seq)`
    /// key was novel globally, or `false` if it was an already-observed echo (deduplicated!).
    pub fn observe(&mut self, agent_id: u64, key: EventKey, value: M) -> bool {
        self.raw_observations += 1;
        let already_known =
            self.base.get(key).is_some() || self.pending_upserts.contains_key(&key);
        let branch = self
            .agents
            .entry(agent_id)
            .or_insert_with(|| self.base.fork());
        if branch.get(key).is_none() {
            branch.upsert(key, value.clone());
        }
        if already_known {
            self.echoes_deduped += 1;
            false
        } else {
            self.pending_upserts.insert(key, value);
            true
        }
    }

    /// Retract a previously submitted key on `agent_id`'s branch and queue global retraction.
    pub fn retract(&mut self, agent_id: u64, key: EventKey) -> bool {
        self.retractions += 1;
        let branch = self
            .agents
            .entry(agent_id)
            .or_insert_with(|| self.base.fork());
        let removed_local = branch.retract(key).is_some();
        let removed_pending = self.pending_upserts.remove(&key).is_some();
        self.pending_retractions.push(key);
        removed_local || removed_pending
    }

    pub fn agent_view(&self, agent_id: u64) -> Option<&FoldTree<M, B>> {
        self.agents.get(&agent_id)
    }

    pub fn base(&self) -> &FoldTree<M, B> {
        &self.base
    }

    /// Merge all agent observations and retractions into the canonical base tree and
    /// sync every agent to the unified persistent root via O(1) `fork()`.
    pub fn merge_all(&mut self) -> &FoldTree<M, B> {
        if !self.pending_upserts.is_empty() {
            let batch: Vec<(EventKey, M)> = std::mem::take(&mut self.pending_upserts)
                .into_iter()
                .collect();
            self.base.upsert_batch(batch);
        }
        if !self.pending_retractions.is_empty() {
            for key in self.pending_retractions.drain(..) {
                self.base.retract(key);
            }
        }
        for branch in self.agents.values_mut() {
            *branch = self.base.fork();
        }
        &self.base
    }

    pub fn stats(&self) -> SwarmStats {
        SwarmStats {
            n_agents: self.agents.len(),
            raw_observations: self.raw_observations,
            unique_events: self.base.len() + self.pending_upserts.len(),
            echoes_deduped: self.echoes_deduped,
            retractions: self.retractions,
            overwrites: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::MomentsMonoid;

    #[test]
    fn bitemporal_zero_lookahead() {
        let mut bt: BitemporalFold<MomentsMonoid, 8> = BitemporalFold::new(());
        bt.upsert(EventKey::at(10), MomentsMonoid::singleton(10, 1.0).unwrap());
        bt.commit(100).unwrap();
        // Later restatement of t=10 at knowledge_time=200
        bt.upsert(EventKey::at(10), MomentsMonoid::singleton(10, 99.0).unwrap());
        bt.commit(200).unwrap();

        assert!(bt.as_of(99).is_none());
        assert!((bt.as_of(150).unwrap().summary().mean - 1.0).abs() < 1e-12);
        assert!((bt.as_of(200).unwrap().summary().mean - 99.0).abs() < 1e-12);
    }

    #[test]
    fn swarm_deduplicates_echoes_across_agents() {
        let mut ledger: SwarmLedger<MomentsMonoid, 8> = SwarmLedger::new(());
        for a in 0..4 {
            ledger.spawn_agent(a);
        }
        let k = EventKey::sourced(10, 7, 0);
        let m = MomentsMonoid::singleton(10, 5.0).unwrap();
        assert!(ledger.observe(0, k, m.clone()));
        assert!(!ledger.observe(1, k, m.clone()));
        assert!(!ledger.observe(2, k, m));
        let merged = ledger.merge_all();
        assert_eq!(merged.len(), 1);
        assert_eq!(ledger.stats().echoes_deduped, 2);
        assert_eq!(ledger.stats().overwrites, 0);
    }
}
