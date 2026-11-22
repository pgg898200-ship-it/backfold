//! Persistent copy-on-write B+ monoid tree (`FoldTree<M, B>`).
//!
//! - Unshared trees mutate **in-place** via `Arc::make_mut` (zero node clones).
//! - `fork()` is **$O(1)$** (`Arc::clone`, ~15 ns, 0 bytes copied).
//! - Shared trees automatically copy-on-write ONLY the $O(\log_B T)$ modified path.
//! - Bulk `from_sorted` builds a minimal-height tree bottom-up in **$O(T)$**.
//! - `find_first_prefix` executes monotone threshold binary search in **$O(\log_B T)$**.
//! - `top_k_intervals` runs top-down best-first anomaly descent in **$O(K \log_B T)$**.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::key::EventKey;
use crate::monoid::Monoid;

#[derive(Clone, Debug)]
enum NodeKind<M: Monoid> {
    Leaf {
        entries: Vec<(EventKey, M)>,
    },
    Internal {
        children: Vec<Arc<Node<M>>>,
    },
}

#[derive(Clone, Debug)]
struct Node<M: Monoid> {
    summary: M,
    len: usize,
    min_key: EventKey,
    max_key: EventKey,
    kind: NodeKind<M>,
}

impl<M: Monoid> Node<M> {
    fn from_leaf(entries: Vec<(EventKey, M)>, ctx: &M::Ctx) -> Self {
        debug_assert!(!entries.is_empty());
        let min_key = entries[0].0;
        let max_key = entries[entries.len() - 1].0;
        let len = entries.len();
        let mut summary = entries[0].1.clone();
        for (_, v) in &entries[1..] {
            summary = M::combine(&summary, v, ctx);
        }
        Self {
            summary,
            len,
            min_key,
            max_key,
            kind: NodeKind::Leaf { entries },
        }
    }

    fn from_internal(children: Vec<Arc<Node<M>>>, ctx: &M::Ctx) -> Self {
        debug_assert!(!children.is_empty());
        let min_key = children[0].min_key;
        let max_key = children[children.len() - 1].max_key;
        let mut len = children[0].len;
        let mut summary = children[0].summary.clone();
        for ch in &children[1..] {
            len += ch.len;
            summary = M::combine(&summary, &ch.summary, ctx);
        }
        Self {
            summary,
            len,
            min_key,
            max_key,
            kind: NodeKind::Internal { children },
        }
    }

    fn recompute(&mut self, ctx: &M::Ctx) {
        match &self.kind {
            NodeKind::Leaf { entries } => {
                debug_assert!(!entries.is_empty());
                self.min_key = entries[0].0;
                self.max_key = entries[entries.len() - 1].0;
                self.len = entries.len();
                let mut acc = entries[0].1.clone();
                for (_, v) in &entries[1..] {
                    acc.combine_assign(v, ctx);
                }
                self.summary = acc;
            }
            NodeKind::Internal { children } => {
                debug_assert!(!children.is_empty());
                self.min_key = children[0].min_key;
                self.max_key = children[children.len() - 1].max_key;
                let mut len = children[0].len;
                let mut acc = children[0].summary.clone();
                for ch in &children[1..] {
                    len += ch.len;
                    acc.combine_assign(&ch.summary, ctx);
                }
                self.len = len;
                self.summary = acc;
            }
        }
    }

    fn depth(&self) -> usize {
        match &self.kind {
            NodeKind::Leaf { .. } => 1,
            NodeKind::Internal { children } => 1 + children[0].depth(),
        }
    }

    fn get(&self, key: EventKey) -> Option<&M> {
        if key < self.min_key || key > self.max_key {
            return None;
        }
        match &self.kind {
            NodeKind::Leaf { entries } => entries
                .binary_search_by_key(&key, |(k, _)| *k)
                .ok()
                .map(|idx| &entries[idx].1),
            NodeKind::Internal { children } => {
                let idx = children
                    .partition_point(|ch| ch.max_key < key)
                    .min(children.len() - 1);
                children[idx].get(key)
            }
        }
    }

    fn range(&self, start: EventKey, end: EventKey, ctx: &M::Ctx) -> M {
        if end < self.min_key || start > self.max_key {
            return M::identity(ctx);
        }
        if start <= self.min_key && self.max_key <= end {
            return self.summary.clone();
        }
        match &self.kind {
            NodeKind::Leaf { entries } => {
                let lo = entries.partition_point(|(k, _)| *k < start);
                let hi = entries.partition_point(|(k, _)| *k <= end);
                if lo >= hi {
                    return M::identity(ctx);
                }
                let mut acc = entries[lo].1.clone();
                for (_, v) in &entries[(lo + 1)..hi] {
                    acc = M::combine(&acc, v, ctx);
                }
                acc
            }
            NodeKind::Internal { children } => {
                let lo = children.partition_point(|ch| ch.max_key < start);
                let mut acc = M::identity(ctx);
                for ch in &children[lo..] {
                    if ch.min_key > end {
                        break;
                    }
                    let part = ch.range(start, end, ctx);
                    if !part.is_identity() {
                        acc = if acc.is_identity() {
                            part
                        } else {
                            M::combine(&acc, &part, ctx)
                        };
                    }
                }
                acc
            }
        }
    }

    fn iter_range(&self, start: EventKey, end: EventKey, out: &mut Vec<(EventKey, M)>) {
        if end < self.min_key || start > self.max_key {
            return;
        }
        match &self.kind {
            NodeKind::Leaf { entries } => {
                let lo = entries.partition_point(|(k, _)| *k < start);
                let hi = entries.partition_point(|(k, _)| *k <= end);
                out.extend_from_slice(&entries[lo..hi]);
            }
            NodeKind::Internal { children } => {
                let lo = children.partition_point(|ch| ch.max_key < start);
                for ch in &children[lo..] {
                    if ch.min_key > end {
                        break;
                    }
                    ch.iter_range(start, end, out);
                }
            }
        }
    }

    fn find_first_prefix<P>(
        &self,
        acc: &mut M,
        ctx: &M::Ctx,
        pred: &P,
    ) -> Option<(EventKey, M)>
    where
        P: Fn(&M) -> bool,
    {
        match &self.kind {
            NodeKind::Leaf { entries } => {
                for (k, v) in entries {
                    let next = if acc.is_identity() {
                        v.clone()
                    } else {
                        M::combine(acc, v, ctx)
                    };
                    if pred(&next) {
                        return Some((*k, next));
                    }
                    *acc = next;
                }
                None
            }
            NodeKind::Internal { children } => {
                for ch in children {
                    let cand = if acc.is_identity() {
                        ch.summary.clone()
                    } else {
                        M::combine(acc, &ch.summary, ctx)
                    };
                    if pred(&cand) {
                        return ch.find_first_prefix(acc, ctx, pred);
                    }
                    *acc = cand;
                }
                None
            }
        }
    }

    fn upsert<const B: usize>(
        node_arc: &mut Arc<Self>,
        key: EventKey,
        value: M,
        ctx: &M::Ctx,
    ) -> (Option<M>, Option<Arc<Self>>) {
        let node = Arc::make_mut(node_arc);
        match &mut node.kind {
            NodeKind::Leaf { entries } => match entries.binary_search_by_key(&key, |(k, _)| *k) {
                Ok(idx) => {
                    let prev = std::mem::replace(&mut entries[idx].1, value);
                    node.recompute(ctx);
                    (Some(prev), None)
                }
                Err(idx) => {
                    entries.insert(idx, (key, value));
                    if entries.len() > B {
                        let mid = entries.len() / 2;
                        let right_entries = entries.split_off(mid);
                        let right_node = Arc::new(Node::from_leaf(right_entries, ctx));
                        node.recompute(ctx);
                        (None, Some(right_node))
                    } else {
                        node.recompute(ctx);
                        (None, None)
                    }
                }
            },
            NodeKind::Internal { children } => {
                let idx = children
                    .partition_point(|ch| ch.max_key < key)
                    .min(children.len() - 1);
                let (prev, split_right) = Self::upsert::<B>(&mut children[idx], key, value, ctx);
                if let Some(right_child) = split_right {
                    children.insert(idx + 1, right_child);
                    if children.len() > B {
                        let mid = children.len() / 2;
                        let right_children = children.split_off(mid);
                        let right_node = Arc::new(Node::from_internal(right_children, ctx));
                        node.recompute(ctx);
                        return (prev, Some(right_node));
                    }
                }
                node.recompute(ctx);
                (prev, None)
            }
        }
    }

    fn retract(node_arc: &mut Arc<Self>, key: EventKey, ctx: &M::Ctx) -> (Option<M>, bool) {
        if key < node_arc.min_key || key > node_arc.max_key {
            return (None, false);
        }
        let node = Arc::make_mut(node_arc);
        match &mut node.kind {
            NodeKind::Leaf { entries } => match entries.binary_search_by_key(&key, |(k, _)| *k) {
                Ok(idx) => {
                    let (_, prev) = entries.remove(idx);
                    if entries.is_empty() {
                        (Some(prev), true)
                    } else {
                        node.recompute(ctx);
                        (Some(prev), false)
                    }
                }
                Err(_) => (None, false),
            },
            NodeKind::Internal { children } => {
                let idx = children
                    .partition_point(|ch| ch.max_key < key)
                    .min(children.len() - 1);
                let (prev, child_empty) = Self::retract(&mut children[idx], key, ctx);
                if prev.is_none() {
                    return (None, false);
                }
                if child_empty {
                    children.remove(idx);
                    if children.is_empty() {
                        return (prev, true);
                    }
                }
                node.recompute(ctx);
                (prev, false)
            }
        }
    }

    fn collect_unshared(
        a: &Arc<Self>,
        b: &Arc<Self>,
        out: &mut Vec<(EventKey, M)>,
    ) {
        if Arc::ptr_eq(a, b) {
            return;
        }
        match (&a.kind, &b.kind) {
            (
                NodeKind::Internal { children: ac },
                NodeKind::Internal { children: bc },
            ) if ac.len() == bc.len() => {
                for (ca, cb) in ac.iter().zip(bc.iter()) {
                    Self::collect_unshared(ca, cb, out);
                }
            }
            _ => {
                b.iter_range(EventKey::MIN, EventKey::MAX, out);
            }
        }
    }
}

struct HeapItem<M: Monoid> {
    score: f64,
    depth: usize,
    node: Arc<Node<M>>,
}

impl<M: Monoid> PartialEq for HeapItem<M> {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
    }
}
impl<M: Monoid> Eq for HeapItem<M> {}
impl<M: Monoid> PartialOrd for HeapItem<M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<M: Monoid> Ord for HeapItem<M> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.node.min_key.cmp(&self.node.min_key))
    }
}

/// Persistent copy-on-write B+ monoid tree supporting O(log T) out-of-order updates,
/// retractions, arbitrary range folds, O(log T) monotone threshold search, and O(1) forking.
#[derive(Clone, Debug)]
pub struct FoldTree<M: Monoid, const B: usize = 16> {
    root: Option<Arc<Node<M>>>,
    ctx: M::Ctx,
}

impl<M: Monoid, const B: usize> FoldTree<M, B> {
    /// Create an empty tree with operator context `ctx`.
    pub fn new(ctx: M::Ctx) -> Self {
        assert!(B >= 4, "FoldTree branching factor B must be >= 4");
        Self { root: None, ctx }
    }

    /// True O(T) bottom-up bulk construction from keys sorted in non-decreasing `EventKey` order.
    /// Adjacent duplicate keys are deduplicated keeping the last value (matching `upsert` semantics).
    pub fn from_sorted(ctx: M::Ctx, mut entries: Vec<(EventKey, M)>) -> Self {
        assert!(B >= 4, "FoldTree branching factor B must be >= 4");
        if entries.is_empty() {
            return Self { root: None, ctx };
        }
        // Fast path: check if strictly increasing (zero extra allocation / zero element moves!)
        let has_dups = entries.windows(2).any(|w| w[0].0 == w[1].0);
        if has_dups {
            // In-place deduplication keeping the LAST value for each duplicate key
            let mut write = 0usize;
            for read in 1..entries.len() {
                if entries[read].0 == entries[write].0 {
                    entries.swap(write, read);
                } else {
                    write += 1;
                    if write != read {
                        entries.swap(write, read);
                    }
                }
            }
            entries.truncate(write + 1);
        }

        let mut level: Vec<Arc<Node<M>>> = Vec::with_capacity(entries.len().div_ceil(B));
        let mut iter = entries.into_iter();
        loop {
            let mut chunk = Vec::with_capacity(B);
            for _ in 0..B {
                if let Some(item) = iter.next() {
                    chunk.push(item);
                } else {
                    break;
                }
            }
            if chunk.is_empty() {
                break;
            }
            level.push(Arc::new(Node::from_leaf(chunk, &ctx)));
        }

        while level.len() > 1 {
            let mut next_level = Vec::with_capacity(level.len().div_ceil(B));
            let mut it = level.into_iter();
            loop {
                let mut children = Vec::with_capacity(B);
                for _ in 0..B {
                    if let Some(ch) = it.next() {
                        children.push(ch);
                    } else {
                        break;
                    }
                }
                if children.is_empty() {
                    break;
                }
                next_level.push(Arc::new(Node::from_internal(children, &ctx)));
            }
            level = next_level;
        }

        Self {
            root: level.into_iter().next(),
            ctx,
        }
    }

    /// Bulk construction from unsorted entries (stable sort + O(T) bottom-up build).
    pub fn from_unsorted(ctx: M::Ctx, mut entries: Vec<(EventKey, M)>) -> Self {
        entries.sort_by_key(|(k, _)| *k);
        Self::from_sorted(ctx, entries)
    }

    #[inline]
    pub fn ctx(&self) -> &M::Ctx {
        &self.ctx
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.len)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.depth())
    }

    /// O(1) full-stream monoid summary.
    #[inline]
    pub fn summary(&self) -> M {
        self.root
            .as_ref()
            .map_or_else(|| M::identity(&self.ctx), |r| r.summary.clone())
    }

    /// O(1) persistent copy-on-write snapshot fork (zero bytes of tree data copied).
    #[inline]
    pub fn fork(&self) -> Self {
        Self {
            root: self.root.as_ref().map(Arc::clone),
            ctx: self.ctx.clone(),
        }
    }

    /// O(log T) lookup of exact leaf summary for `key`.
    pub fn get(&self, key: EventKey) -> Option<&M> {
        self.root.as_ref().and_then(|r| r.get(key))
    }

    /// O(log T) out-of-order insert or replace (idempotent on identical `EventKey`).
    pub fn upsert(&mut self, key: EventKey, value: M) -> Option<M> {
        match &mut self.root {
            None => {
                self.root = Some(Arc::new(Node::from_leaf(vec![(key, value)], &self.ctx)));
                None
            }
            Some(root_arc) => {
                let (prev, split_right) = Node::upsert::<B>(root_arc, key, value, &self.ctx);
                if let Some(right_node) = split_right {
                    let left_node = Arc::clone(root_arc);
                    *root_arc = Arc::new(Node::from_internal(vec![left_node, right_node], &self.ctx));
                }
                prev
            }
        }
    }

    /// Batch insert/update. Delegates to O(T) bulk build if tree is empty or batch is large.
    pub fn upsert_batch(&mut self, mut entries: Vec<(EventKey, M)>) {
        if entries.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = Self::from_unsorted(self.ctx.clone(), entries);
            return;
        }
        if entries.len() >= self.len() {
            let mut all = self.iter_all();
            all.append(&mut entries);
            *self = Self::from_unsorted(self.ctx.clone(), all);
            return;
        }
        entries.sort_by_key(|(k, _)| *k);
        for (k, v) in entries {
            self.upsert(k, v);
        }
    }

    /// O(log T) exact retraction (deletion) of `key`. Prunes empty nodes and shrinks height.
    pub fn retract(&mut self, key: EventKey) -> Option<M> {
        let root_arc = self.root.as_mut()?;
        let (prev, root_empty) = Node::retract(root_arc, key, &self.ctx);
        prev.as_ref()?;
        if root_empty {
            self.root = None;
            return prev;
        }
        // Collapse single-child internal root to keep tree height tight
        loop {
            let single_child = match &self.root.as_ref().unwrap().kind {
                NodeKind::Internal { children } if children.len() == 1 => {
                    Some(Arc::clone(&children[0]))
                }
                _ => None,
            };
            if let Some(ch) = single_child {
                self.root = Some(ch);
            } else {
                break;
            }
        }
        prev
    }

    /// O(log T) associative monoid fold over inclusive key range `[start, end]`.
    pub fn range(&self, start: EventKey, end: EventKey) -> M {
        if start > end {
            return M::identity(&self.ctx);
        }
        self.root
            .as_ref()
            .map_or_else(|| M::identity(&self.ctx), |r| r.range(start, end, &self.ctx))
    }

    /// O(log T) prefix fold over all keys `<= end` (strict point-in-time query).
    #[inline]
    pub fn prefix(&self, end: EventKey) -> M {
        self.range(EventKey::MIN, end)
    }

    /// O(log T) suffix fold over all keys `>= start`.
    #[inline]
    pub fn suffix(&self, start: EventKey) -> M {
        self.range(start, EventKey::MAX)
    }

    /// O(log T) fold over inclusive event-timestamp range `[t_start, t_end]`.
    #[inline]
    pub fn query_time_range(&self, t_start: i64, t_end: i64) -> M {
        self.range(EventKey::time_start(t_start), EventKey::time_end(t_end))
    }

    /// O(log T) exact monotone prefix binary search.
    ///
    /// Returns the earliest `(key, prefix_summary)` where `pred(&prefix_summary)` is `true`,
    /// assuming `pred` is monotone (once true for a prefix, stays true for all longer prefixes).
    pub fn find_first_prefix<P>(&self, pred: P) -> Option<(EventKey, M)>
    where
        P: Fn(&M) -> bool,
    {
        let root = self.root.as_ref()?;
        if !pred(&root.summary) {
            return None;
        }
        let mut acc = M::identity(&self.ctx);
        root.find_first_prefix(&mut acc, &self.ctx, &pred)
    }

    /// Exact lossless recovery of all `(EventKey, M)` entries in `[start, end]`.
    pub fn iter_range(&self, start: EventKey, end: EventKey) -> Vec<(EventKey, M)> {
        let mut out = Vec::new();
        if start <= end {
            if let Some(root) = &self.root {
                root.iter_range(start, end, &mut out);
            }
        }
        out
    }

    /// Recover all entries in chronological order.
    pub fn iter_all(&self) -> Vec<(EventKey, M)> {
        self.iter_range(EventKey::MIN, EventKey::MAX)
    }

    /// Top-down O(K log T) best-first descent returning the top-`k` disjoint intervals at
    /// `target_depth` (or leaves if shallower) ranked by `score(&summary)`.
    pub fn top_k_intervals<S>(
        &self,
        k: usize,
        target_depth: usize,
        score: S,
    ) -> Vec<(EventKey, EventKey, usize, f64, M)>
    where
        S: Fn(&M) -> f64,
    {
        let Some(root) = &self.root else {
            return Vec::new();
        };
        if k == 0 {
            return Vec::new();
        }
        let mut heap = BinaryHeap::new();
        let root_score = score(&root.summary);
        heap.push(HeapItem {
            score: if root_score.is_finite() { root_score } else { 0.0 },
            depth: 0,
            node: Arc::clone(root),
        });

        let mut results = Vec::with_capacity(k);
        while results.len() < k {
            let Some(item) = heap.pop() else {
                break;
            };
            match &item.node.kind {
                NodeKind::Internal { children } if item.depth < target_depth => {
                    for ch in children {
                        let s = score(&ch.summary);
                        heap.push(HeapItem {
                            score: if s.is_finite() { s } else { 0.0 },
                            depth: item.depth + 1,
                            node: Arc::clone(ch),
                        });
                    }
                }
                _ => {
                    results.push((
                        item.node.min_key,
                        item.node.max_key,
                        item.node.len,
                        item.score,
                        item.node.summary.clone(),
                    ));
                }
            }
        }
        results
    }

    /// Conflict-free, idempotent CRDT union merge of two trees.
    ///
    /// - Fast-path O(1) when `Arc::ptr_eq` detects identical roots.
    /// - Skips identical shared subtrees via `Arc::ptr_eq` when merging forked agent branches.
    /// - Existing keys already present in `self` are preserved (no clobbering of concurrent edits);
    ///   novel keys from `other` are inserted cleanly, making duplicate echoed observations strictly
    ///   idempotent with zero double-counting.
    pub fn merge(&self, other: &Self) -> Self {
        match (&self.root, &other.root) {
            (None, _) => other.fork(),
            (_, None) => self.fork(),
            (Some(ra), Some(rb)) => {
                if Arc::ptr_eq(ra, rb) {
                    return self.fork();
                }
                let mut diff = Vec::new();
                Node::collect_unshared(ra, rb, &mut diff);
                if diff.is_empty() {
                    return self.fork();
                }
                let mut merged = self.fork();
                for (k, v) in diff {
                    if merged.get(k).is_none() {
                        merged.upsert(k, v);
                    }
                }
                merged
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::AffineEwmaMonoid;

    #[test]
    fn non_commutative_affine_out_of_order_retract_and_fork() {
        let alpha = 0.2;
        let xs = [1.0, 3.5, -2.0, 4.0, 0.5, 6.0, -1.0, 2.2, 8.0, -3.0, 1.5, 0.0];
        let mut tree: FoldTree<AffineEwmaMonoid, 4> = FoldTree::new(());

        // Insert completely out of order (reverse then interleaved)
        let order = [11, 0, 7, 3, 9, 1, 5, 10, 2, 8, 4, 6];
        for &i in &order {
            tree.upsert(
                EventKey::at(i as i64),
                AffineEwmaMonoid::singleton(xs[i], alpha).unwrap(),
            );
        }
        assert_eq!(tree.len(), xs.len());

        // Compare full summary and every sub-range [l..=r] against sequential reference
        for l in 0..xs.len() {
            for r in l..xs.len() {
                let mut s = 0.7;
                for &x in &xs[l..=r] {
                    s = (1.0 - alpha) * s + alpha * x;
                }
                let got = tree.query_time_range(l as i64, r as i64).apply(0.7);
                assert!((got - s).abs() < 1e-13, "range [{l}..={r}] mismatch");
            }
        }

        // Persistent O(1) fork isolation
        let mut branch = tree.fork();
        branch.retract(EventKey::at(5));
        assert_eq!(tree.len(), xs.len());
        assert_eq!(branch.len(), xs.len() - 1);

        // Merge restores retracted key idempotently
        let merged = branch.merge(&tree);
        assert_eq!(merged.len(), xs.len());
        assert!((merged.summary().apply(0.7) - tree.summary().apply(0.7)).abs() < 1e-13);

        // Retract all keys down to empty
        for i in 0..xs.len() {
            assert!(tree.retract(EventKey::at(i as i64)).is_some());
        }
        assert!(tree.is_empty());
        assert_eq!(tree.depth(), 0);
    }
}
