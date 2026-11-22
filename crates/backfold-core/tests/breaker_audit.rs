//! Independent Adversarial Breaker Audit for `backfold-core`.
//!
//! Actively attempts to break every monoid, tree invariant, numerical edge case,
//! statistical FDR guarantee (including heavy-tailed Student-t3 and Cauchy noise),
//! and multi-threaded persistent fork/merge isolation invariant.

use std::collections::BTreeMap;
use std::thread;

use backfold_core::ops::{
    ebh_reject, smooth_1d, smooth_2d, smooth_nd, CovarianceMonoid, CusumMonoid, EProcessMonoid,
    EwmaMonoid, KalmanFilter1D, KalmanFilter2D, KalmanMonoid, MomentsMonoid, RidgeMonoid,
};
use backfold_core::{BackfoldError, EventKey, FoldTree, Monoid};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn rand_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn rand_cauchy(rng: &mut StdRng) -> f64 {
    let u: f64 = rng.gen::<f64>().clamp(1e-7, 1.0 - 1e-7);
    (std::f64::consts::PI * (u - 0.5)).tan()
}

fn rand_student_t3(rng: &mut StdRng) -> f64 {
    let z = rand_normal(rng);
    let chi2 = rand_normal(rng).powi(2) + rand_normal(rng).powi(2) + rand_normal(rng).powi(2);
    z / (chi2 / 3.0).sqrt().max(1e-9)
}

// 1. 500 random out-of-order upserts, overwrites, retractions (including to empty),
// and re-insertions across ALL monoids vs independent naive sequential reference folds.
#[test]
fn random_permutation_retraction_and_reinsert_all_monoids() {
    let mut rng = StdRng::seed_from_u64(0xBADC_0FFE_1234_5678);

    // Moments, EWMA, CUSUM, EProcess, Kalman1D, Kalman2D, Ridge5 (inline), Ridge10 (heap)
    let lam = 0.015;
    let mut t_mom: FoldTree<MomentsMonoid, 16> = FoldTree::new(());
    let mut t_ewma: FoldTree<EwmaMonoid, 16> = FoldTree::new(lam);
    let mut t_cus: FoldTree<CusumMonoid, 16> = FoldTree::new(());
    let mut t_ep: FoldTree<EProcessMonoid, 16> = FoldTree::new(());
    let mut t_k1: FoldTree<KalmanFilter1D, 16> = FoldTree::new(());
    let mut t_k2: FoldTree<KalmanFilter2D, 4> = FoldTree::new(());
    let mut t_r5: FoldTree<RidgeMonoid, 4> = FoldTree::new((5, lam));
    let mut t_r10: FoldTree<RidgeMonoid, 4> = FoldTree::new((10, lam));

    let mut truth_x: BTreeMap<EventKey, f64> = BTreeMap::new();
    let mut truth_r5: BTreeMap<EventKey, ([f64; 5], f64)> = BTreeMap::new();
    let mut truth_r10: BTreeMap<EventKey, ([f64; 10], f64)> = BTreeMap::new();

    for step in 0..500 {
        let op = rng.gen_range(0..10);
        if op < 7 || truth_x.is_empty() {
            let ts = rng.gen_range(-200i64..200i64);
            let src = rng.gen_range(0u64..3u64);
            let key = EventKey::sourced(ts, src, 0);
            let x = rand_normal(&mut rng) * 3.0;
            let mut f5 = [0.0; 5];
            let mut f10 = [0.0; 10];
            for v in &mut f5 {
                *v = rand_normal(&mut rng);
            }
            for v in &mut f10 {
                *v = rand_normal(&mut rng);
            }
            let y = 0.8 * f5[0] - 0.5 * f5[1] + 0.2 * rand_normal(&mut rng);

            truth_x.insert(key, x);
            truth_r5.insert(key, (f5, y));
            truth_r10.insert(key, (f10, y));

            t_mom.upsert(key, MomentsMonoid::singleton(ts, x).unwrap());
            t_ewma.upsert(key, EwmaMonoid::singleton(ts, x).unwrap());
            t_cus.upsert(key, CusumMonoid::singleton(ts, x, 0.0, 3.0, 0.5).unwrap());
            t_ep.upsert(key, EProcessMonoid::singleton_z(ts, x / 3.0).unwrap());
            t_k1.upsert(key, KalmanFilter1D::from_random_walk(0.5, 0.1, 0.4, x).unwrap());
            t_k2.upsert(
                key,
                KalmanFilter2D::from_constant_velocity(0.5, 0.05, 0.4, x).unwrap(),
            );
            t_r5.upsert(key, RidgeMonoid::singleton(ts, &f5, y, 1.0).unwrap());
            t_r10.upsert(key, RidgeMonoid::singleton(ts, &f10, y, 1.0).unwrap());
        } else {
            let keys: Vec<_> = truth_x.keys().copied().collect();
            let victim = keys[rng.gen_range(0..keys.len())];
            truth_x.remove(&victim);
            truth_r5.remove(&victim);
            truth_r10.remove(&victim);

            assert!(t_mom.retract(victim).is_some());
            assert!(t_ewma.retract(victim).is_some());
            assert!(t_cus.retract(victim).is_some());
            assert!(t_ep.retract(victim).is_some());
            assert!(t_k1.retract(victim).is_some());
            assert!(t_k2.retract(victim).is_some());
            assert!(t_r5.retract(victim).is_some());
            assert!(t_r10.retract(victim).is_some());
        }

        // Periodically verify exact sequential fold parity
        if step % 50 == 49 {
            let mut ref_mom = MomentsMonoid::identity(&());
            let mut ref_ewma = EwmaMonoid::identity(&lam);
            let mut ref_cus = CusumMonoid::identity(&());
            let mut ref_ep = EProcessMonoid::identity(&());
            let mut ref_k1 = KalmanFilter1D::identity(&());
            let mut ref_k2 = KalmanFilter2D::identity(&());
            for (&k, &x) in &truth_x {
                ref_mom = MomentsMonoid::combine(
                    &ref_mom,
                    &MomentsMonoid::singleton(k.event_time, x).unwrap(),
                    &(),
                );
                ref_ewma = EwmaMonoid::combine(
                    &ref_ewma,
                    &EwmaMonoid::singleton(k.event_time, x).unwrap(),
                    &lam,
                );
                ref_cus = CusumMonoid::combine(
                    &ref_cus,
                    &CusumMonoid::singleton(k.event_time, x, 0.0, 3.0, 0.5).unwrap(),
                    &(),
                );
                ref_ep = EProcessMonoid::combine(
                    &ref_ep,
                    &EProcessMonoid::singleton_z(k.event_time, x / 3.0).unwrap(),
                    &(),
                );
                ref_k1 = KalmanFilter1D::combine(
                    &ref_k1,
                    &KalmanFilter1D::from_random_walk(0.5, 0.1, 0.4, x).unwrap(),
                    &(),
                );
                ref_k2 = KalmanFilter2D::combine(
                    &ref_k2,
                    &KalmanFilter2D::from_constant_velocity(0.5, 0.05, 0.4, x).unwrap(),
                    &(),
                );
            }
            let gm = t_mom.summary();
            assert_eq!(gm.n, ref_mom.n);
            assert!((gm.mean - ref_mom.mean).abs() < 1e-11);
            assert!((gm.variance() - ref_mom.variance()).abs() < 1e-10);
            assert!((gm.skewness() - ref_mom.skewness()).abs() < 1e-10);
            assert!((gm.excess_kurtosis() - ref_mom.excess_kurtosis()).abs() < 1e-10);

            let ge = t_ewma.summary();
            assert!((ge.mean - ref_ewma.mean).abs() < 1e-11);
            assert!((ge.variance() - ref_ewma.variance()).abs() < 1e-10);

            let gc = t_cus.summary();
            assert!((gc.max_peak() - ref_cus.max_peak()).abs() < 1e-11);
            assert!((gc.current_pos() - ref_cus.current_pos()).abs() < 1e-11);

            let gep = t_ep.summary();
            assert!((gep.log_wealth() - ref_ep.log_wealth()).abs() < 1e-11);
            assert!((gep.log_e_detector() - ref_ep.log_e_detector()).abs() < 1e-11);
            assert!((gep.sr_evalue() - ref_ep.sr_evalue()).abs() <= 1e-10 * ref_ep.sr_evalue().max(1.0));
            assert!(
                (gep.scan_evalue() - ref_ep.scan_evalue()).abs()
                    <= 1e-10 * ref_ep.scan_evalue().max(1.0)
            );

            let (m1a, p1a) = t_k1.summary().posterior(0.0, 5.0).unwrap();
            let (m1b, p1b) = ref_k1.posterior(0.0, 5.0).unwrap();
            assert!((m1a - m1b).abs() < 1e-11 && (p1a - p1b).abs() < 1e-11);

            let (m2a, p2a) = t_k2
                .summary()
                .posterior([0.0, 0.0], [5.0, 0.0, 0.0, 5.0])
                .unwrap();
            let (m2b, p2b) = ref_k2
                .posterior([0.0, 0.0], [5.0, 0.0, 0.0, 5.0])
                .unwrap();
            for i in 0..2 {
                assert!((m2a[i] - m2b[i]).abs() < 1e-10);
            }
            for i in 0..4 {
                assert!((p2a[i] - p2b[i]).abs() < 1e-10);
            }

            // Ridge inline (dim=5) and heap (dim=10)
            let mut ref_r5 = RidgeMonoid::identity(&(5, lam));
            for (&k, &(f, y)) in &truth_r5 {
                ref_r5 = RidgeMonoid::combine(
                    &ref_r5,
                    &RidgeMonoid::singleton(k.event_time, &f, y, 1.0).unwrap(),
                    &(5, lam),
                );
            }
            let s5a = t_r5.summary().solve(0.1).unwrap();
            let s5b = ref_r5.solve(0.1).unwrap();
            for i in 0..5 {
                assert!((s5a.weights[i] - s5b.weights[i]).abs() < 1e-10);
            }

            let mut ref_r10 = RidgeMonoid::identity(&(10, lam));
            for (&k, &(f, y)) in &truth_r10 {
                ref_r10 = RidgeMonoid::combine(
                    &ref_r10,
                    &RidgeMonoid::singleton(k.event_time, &f, y, 1.0).unwrap(),
                    &(10, lam),
                );
            }
            let s10a = t_r10.summary().solve(0.1).unwrap();
            let s10b = ref_r10.solve(0.1).unwrap();
            for i in 0..10 {
                assert!((s10a.weights[i] - s10b.weights[i]).abs() < 1e-10);
            }
        }
    }

    // Retract ALL remaining keys down to empty and verify clean identity + re-insertion
    let rem: Vec<_> = truth_x.keys().copied().collect();
    for k in rem {
        assert!(t_mom.retract(k).is_some());
    }
    assert!(t_mom.is_empty());
    assert_eq!(t_mom.summary().n, 0);
    t_mom.upsert(EventKey::at(42), MomentsMonoid::singleton(42, 7.5).unwrap());
    assert_eq!(t_mom.len(), 1);
    assert!((t_mom.summary().mean - 7.5).abs() < 1e-15);

    // Also verify general dim=3 KalmanMonoid with obs_dim=2 vector obs + missing step
    let f3 = [1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.9];
    let q3 = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05];
    let h23 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let r22 = [0.2, 0.02, 0.02, 0.3];
    let mut tk3: FoldTree<KalmanMonoid, 4> = FoldTree::new(3);
    let mut seq_k3 = KalmanMonoid::identity(&3);
    for t in 0..40i64 {
        let leaf = if t % 7 == 0 {
            KalmanMonoid::from_missing(3, &f3, &q3).unwrap()
        } else {
            let y = [t as f64 * 0.1 + 0.2, -0.3 + (t as f64 * 0.2).sin()];
            KalmanMonoid::from_vector_obs(3, 2, &f3, &q3, &h23, &r22, &y).unwrap()
        };
        seq_k3 = KalmanMonoid::combine(&seq_k3, &leaf, &3);
        tk3.upsert(EventKey::at(t), leaf);
    }
    let m0 = [0.0, 0.0, 0.0];
    let p0 = [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0];
    let p_tree = tk3.summary().posterior(&m0, &p0).unwrap();
    let p_seq = seq_k3.posterior(&m0, &p0).unwrap();
    for i in 0..3 {
        assert!((p_tree.mean[i] - p_seq.mean[i]).abs() < 1e-11);
    }
    for i in 0..9 {
        assert!((p_tree.cov[i] - p_seq.cov[i]).abs() < 1e-11);
    }
}

// 2. Extreme numerical torture: 1e12 offset cancellation, 1e150 spike + eviction,
// huge dt = 1e12, negative timestamps, and NaN/Inf rejection.
#[test]
fn extreme_numerical_torture() {
    let mut rng = StdRng::seed_from_u64(999);
    // 1e12 + N(0, 1) offset cancellation: verify exact parity (<1e-12) against the true
    // residuals actually represented in the IEEE-754 f64 inputs `val = base + d`.
    let base = 1e12_f64;
    let mut tm: FoldTree<MomentsMonoid, 16> = FoldTree::new(());
    let mut centered = Vec::with_capacity(300);
    for i in 0..300i64 {
        let d = rand_normal(&mut rng);
        let val = base + d;
        centered.push(val - base);
        tm.upsert(EventKey::at(i), MomentsMonoid::singleton(i, val).unwrap());
    }
    let c_mean = centered.iter().sum::<f64>() / 300.0;
    let c_var = centered.iter().map(|&x| (x - c_mean).powi(2)).sum::<f64>() / 300.0;
    assert!((tm.summary().variance() - c_var).abs() < 1e-12);

    // 1e150 spike inserted then retracted -> ZERO residual corruption!
    tm.upsert(EventKey::at(150), MomentsMonoid::singleton(150, 1e150).unwrap());
    assert!(tm.summary().max >= 1e149);
    tm.upsert(
        EventKey::at(150),
        MomentsMonoid::singleton(150, base + centered[150]).unwrap(),
    );
    assert!((tm.summary().variance() - c_var).abs() < 1e-12);

    // Huge timestamps & dt = 1e12 in EWMA & Ridge (subnormal/underflow immunity)
    let mut te: FoldTree<EwmaMonoid, 16> = FoldTree::new(0.1);
    te.upsert(EventKey::at(-1_000_000_000_000), EwmaMonoid::singleton(-1_000_000_000_000, 100.0).unwrap());
    te.upsert(EventKey::at(1_000_000_000_000), EwmaMonoid::singleton(1_000_000_000_000, 2.0).unwrap());
    let s = te.summary();
    assert!(s.mean.is_finite() && (s.mean - 2.0).abs() < 1e-12 && s.variance() == 0.0);

    // NaN / Inf / -Inf rejection across operators
    assert!(matches!(MomentsMonoid::singleton(0, f64::NAN), Err(BackfoldError::NonFiniteInput(_))));
    assert!(matches!(MomentsMonoid::singleton(0, f64::INFINITY), Err(BackfoldError::NonFiniteInput(_))));
    assert!(matches!(EwmaMonoid::singleton(0, f64::NEG_INFINITY), Err(BackfoldError::NonFiniteInput(_))));
    assert!(matches!(CusumMonoid::singleton_z(0, f64::NAN, 0.5), Err(BackfoldError::NonFiniteInput(_))));
    assert!(matches!(EProcessMonoid::singleton_z(0, f64::INFINITY), Err(BackfoldError::NonFiniteInput(_))));
    assert!(matches!(
        KalmanFilter2D::from_constant_velocity(1.0, 0.1, 0.5, f64::NAN),
        Err(BackfoldError::NonFiniteInput(_))
    ));
    assert!(matches!(
        RidgeMonoid::singleton(0, &[1.0, f64::NAN], 1.0, 1.0),
        Err(BackfoldError::NonFiniteInput(_))
    ));
}

/// 3. CUSUM first-crossing exact boundary, negative-only crossing, and retraction behavior.
#[test]
fn cusum_first_crossing_exact_adversarial() {
    let mut tree: FoldTree<CusumMonoid, 8> = FoldTree::new(());
    assert!(tree.find_first_prefix(|m| m.max_peak() >= 1.0).is_none());

    // Single-element exact equality boundary: z = 5.5, k = 0.5 -> S_pos = 5.0 == threshold
    tree.upsert(EventKey::at(10), CusumMonoid::singleton_z(10, 5.5, 0.5).unwrap());
    assert_eq!(
        tree.find_first_prefix(|m| m.max_peak() >= 5.0).map(|(k, _)| k.event_time),
        Some(10)
    );
    assert!(tree.find_first_prefix(|m| m.max_peak() >= 5.0001).is_none());

    // Retracting the only crossing returns None
    tree.retract(EventKey::at(10));
    assert!(tree.find_first_prefix(|m| m.max_peak() >= 5.0).is_none());

    // Negative-only crossing at t = 25..29
    for t in 0..50i64 {
        let z = if (25..30).contains(&t) { -3.0 } else { 0.0 };
        tree.upsert(EventKey::at(t), CusumMonoid::singleton_z(t, z, 0.5).unwrap());
    }
    // Each negative tick adds (-(-3.0) - 0.5) = 2.5 -> t=25: 2.5, t=26: 5.0, t=27: 7.5, t=28: 10.0
    assert_eq!(
        tree.find_first_prefix(|m| m.max_peak() >= 10.0).map(|(k, _)| k.event_time),
        Some(28)
    );
    // Retracting t=26 moves threshold=10.0 crossing to t=29!
    tree.retract(EventKey::at(26));
    assert_eq!(
        tree.find_first_prefix(|m| m.max_peak() >= 10.0).map(|(k, _)| k.event_time),
        Some(29)
    );
}

/// 4. Ill-conditioned (cond ~ 1e24 without equilibration) and collinear rank-deficient Ridge.
#[test]
fn ill_conditioned_and_rank_deficient_ridge() {
    let mut rng = StdRng::seed_from_u64(4242);
    let ctx = (4usize, 0.0f64);
    let mut tree: FoldTree<RidgeMonoid, 4> = FoldTree::new(ctx);
    for t in 0..200i64 {
        let a = rand_normal(&mut rng);
        let b = rand_normal(&mut rng);
        // Col 0 = 1e6 * a, Col 1 = 1e-6 * b, Col 2 == Col 0 (rank-deficient collinear!), Col 3 = 1.0
        let x = [1e6 * a, 1e-6 * b, 1e6 * a, 1.0];
        let y = 2.0 * a - 3.0 * b + 0.5;
        tree.upsert(EventKey::at(t), RidgeMonoid::singleton(t, &x, y, 1.0).unwrap());
    }
    // Note: Col 1 has G[1,1] ~ 2e-10, so alpha = 1e-15 keeps Ridge shrinkage < 1e-5
    // while alpha is still 29 orders of magnitude below G[0,0] ~ 2e14 (testing equilibrated jitter!).
    let sol = tree.summary().solve(1e-15).unwrap();
    assert!(sol.weights.iter().all(|w| w.is_finite()));
    assert!(sol.residual_variance.is_finite() && sol.residual_variance >= 0.0);
    let (yhat, pvar) = tree
        .summary()
        .predict_with_variance(&[1e6, 1e-6, 1e6, 1.0], 1e-15)
        .unwrap();
    // True y at a=1, b=1 is 2 - 3 + 0.5 = -0.5
    assert!((yhat - (-0.5)).abs() < 1e-4, "yhat={yhat}");
    assert!(pvar.is_finite() && pvar >= 0.0);
}

/// 5. Kalman extreme R/Q ratios (1e-6..1e6), 2D fast-path parity vs general KalmanMonoid, and PSD invariants.
#[test]
fn kalman_extreme_rq_and_psd_invariants() {
    let mut rng = StdRng::seed_from_u64(7777);
    for &(q_acc, r_obs) in &[(1e-6, 1.0), (1.0, 1e-6), (1e-4, 1e2), (1e2, 1e-4)] {
        let mut t_fast: FoldTree<KalmanFilter2D, 4> = FoldTree::new(());
        let mut t_gen: FoldTree<KalmanMonoid, 4> = FoldTree::new(2);
        for t in 0..100i64 {
            let dt = rng.gen_range(0.1..2.0);
            let y = (t as f64) * 0.05 + rand_normal(&mut rng);
            t_fast.upsert(
                EventKey::at(t),
                KalmanFilter2D::from_constant_velocity(dt, q_acc, r_obs, y).unwrap(),
            );
            t_gen.upsert(
                EventKey::at(t),
                KalmanMonoid::from_constant_velocity(dt, q_acc, r_obs, y).unwrap(),
            );
        }
        let m0 = [0.0, 0.0];
        let p0 = [10.0, 0.0, 0.0, 10.0];
        let (mf, pf) = t_fast.summary().posterior(m0, p0).unwrap();
        let pg = t_gen.summary().posterior(&m0, &p0).unwrap();
        for i in 0..2 {
            assert!((mf[i] - pg.mean[i]).abs() < 1e-10, "mean diff at ({q_acc},{r_obs})");
        }
        for i in 0..4 {
            assert!((pf[i] - pg.cov[i]).abs() < 1e-10, "cov diff at ({q_acc},{r_obs})");
        }
        // Symmetric PSD checks
        assert!(pf[0] >= 0.0 && pf[3] >= 0.0);
        assert!((pf[1] - pf[2]).abs() < 1e-14);
        assert!(pf[0] * pf[3] - pf[1] * pf[2] >= -1e-12);
    }
}

// 6. Monte Carlo empirical FDR verification (200 trials x 40 streams) under Gaussian,
// heavy-tailed Student-t(df=3), AND Cauchy (infinite-variance!) noise at alpha = 0.05.
#[test]
fn monte_carlo_fdr_control_gaussian_and_heavy_tail() {
    let mut rng = StdRng::seed_from_u64(20_261_122);
    let alpha = 0.05;
    let n_trials = 200;
    let m_streams = 40;
    let n_true = 5;
    let t_len = 160;

    for dist_name in ["gaussian", "student_t3", "cauchy"] {
        let mut fdp_sum = 0.0;
        let mut recall_sum = 0.0;
        for _ in 0..n_trials {
            let mut e_vals = Vec::with_capacity(m_streams);
            for s in 0..m_streams {
                let mut acc = EProcessMonoid::identity(&());
                for t in 0..t_len {
                    let noise = match dist_name {
                        "gaussian" => rand_normal(&mut rng),
                        "student_t3" => rand_student_t3(&mut rng),
                        _ => rand_cauchy(&mut rng),
                    };
                    let shift = if s < n_true && t >= 60 { 2.5 } else { 0.0 };
                    let leaf = EProcessMonoid::singleton_z(t as i64, noise + shift).unwrap();
                    acc = EProcessMonoid::combine(&acc, &leaf, &());
                }
                e_vals.push(acc.e_value());
            }
            let rej = ebh_reject(&e_vals, alpha);
            let tp = rej.iter().filter(|&&idx| idx < n_true).count();
            let fp = rej.len() - tp;
            if !rej.is_empty() {
                fdp_sum += fp as f64 / rej.len() as f64;
            }
            recall_sum += tp as f64 / n_true as f64;
        }
        let emp_fdr = fdp_sum / n_trials as f64;
        let emp_recall = recall_sum / n_trials as f64;
        assert!(
            emp_fdr <= 0.06,
            "{dist_name}: empirical FDR {emp_fdr:.4} exceeded 0.06 (nominal alpha={alpha})"
        );
        assert!(
            emp_recall >= 0.80,
            "{dist_name}: empirical recall {emp_recall:.4} too low"
        );
    }
}

/// 7. Multi-threaded concurrent persistent fork, upsert, retract, and idempotent CRDT merge isolation.
#[test]
fn multithreaded_fork_and_merge_isolation() {
    let entries: Vec<_> = (0..5_000i64)
        .map(|i| (EventKey::at(i), MomentsMonoid::singleton(i, (i % 17) as f64).unwrap()))
        .collect();
    let root: FoldTree<MomentsMonoid, 16> = FoldTree::from_sorted((), entries);
    let orig_summary = root.summary();

    let handles: Vec<_> = (0..8u64)
        .map(|tid| {
            let mut branch = root.fork();
            thread::spawn(move || {
                for j in 0..200i64 {
                    let ts = 10_000 + (tid as i64) * 1_000 + j;
                    branch.upsert(
                        EventKey::sourced(ts, tid + 1, 0),
                        MomentsMonoid::singleton(ts, (tid as f64) * 10.0 + (j as f64)).unwrap(),
                    );
                }
                branch
            })
        })
        .collect();

    let branches: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Original root MUST be 100% untouched (strict persistent snapshot isolation)
    assert_eq!(root.len(), 5_000);
    assert_eq!(root.summary(), orig_summary);

    // Merge forward vs merge reverse -> identical!
    let mut fwd = root.fork();
    for b in &branches {
        fwd = fwd.merge(b);
    }
    let mut rev = root.fork();
    for b in branches.iter().rev() {
        rev = rev.merge(b);
    }
    // Idempotent double-merge
    let fwd_twice = fwd.merge(&branches[0]).merge(&branches[3]);

    assert_eq!(fwd.len(), 5_000 + 8 * 200);
    assert_eq!(rev.len(), fwd.len());
    assert_eq!(fwd_twice.len(), fwd.len());
    assert!((fwd.summary().mean - rev.summary().mean).abs() < 1e-12);
    assert!((fwd.summary().variance() - rev.summary().variance()).abs() < 1e-12);
    assert!((fwd.summary().mean - fwd_twice.summary().mean).abs() < 1e-14);
}

// 8. Adversarial RTS Smoother + Exact Marginal Log-Likelihood across 1D, 2D, and 3D vector-obs
// with out-of-order upserts, missing observations, retractions, boundary timestamps, and PSD
// variance-reduction invariant (P_{t|T} <= P_{t|t}).
#[test]
fn rts_smoother_and_marginal_log_likelihood_adversarial() {
    let mut rng = StdRng::seed_from_u64(0x5100_74ED);

    // 1D exact parity with sequential RTS + innovation log-likelihood after late edit & retraction
    let n = 60usize;
    let q = 0.09;
    let r = 0.45;
    let m0 = 0.7;
    let p0 = 5.0;
    let mut ys: Vec<Option<f64>> = (0..n)
        .map(|i| {
            if i % 9 == 4 {
                None
            } else {
                Some(0.1 * (i as f64) + rand_normal(&mut rng))
            }
        })
        .collect();

    let mut t1: FoldTree<KalmanFilter1D, 8> = FoldTree::new(());
    // Insert out of order
    for i in (0..n).rev() {
        let leaf = match ys[i] {
            Some(y) => KalmanFilter1D::from_random_walk(1.0, q, r, y).unwrap(),
            None => KalmanFilter1D::from_missing(1.0, q).unwrap(),
        };
        t1.upsert(EventKey::at((i + 1) as i64), leaf);
    }
    // Overwrite a middle point & retract a point
    ys[20] = Some(12.5);
    t1.upsert(
        EventKey::at(21),
        KalmanFilter1D::from_random_walk(1.0, q, r, 12.5).unwrap(),
    );
    t1.retract(EventKey::at(35));

    // Sequential 1D forward + RTS backward over active keys
    let mut active_ts = Vec::new();
    let mut mf = vec![m0];
    let mut pf = vec![p0];
    let mut mp = Vec::new();
    let mut pp = Vec::new();
    let mut seq_ll = 0.0;
    const LN_TAU: f64 = 1.837_877_066_409_345_3;
    for (i, &yo) in ys.iter().enumerate() {
        let ts = (i + 1) as i64;
        if ts == 35 {
            continue;
        }
        active_ts.push(ts);
        let m_p = *mf.last().unwrap();
        let p_p = *pf.last().unwrap() + q;
        mp.push(m_p);
        pp.push(p_p);
        if let Some(y) = yo {
            let s = p_p + r;
            let innov = y - m_p;
            seq_ll += -0.5 * (LN_TAU + s.ln() + innov * innov / s);
            let k = p_p / s;
            mf.push(m_p + k * innov);
            pf.push((1.0 - k) * p_p);
        } else {
            mf.push(m_p);
            pf.push(p_p);
        }
    }
    let ll1 = t1.summary().log_marginal_likelihood(m0, p0).unwrap();
    assert!((ll1 - seq_ll).abs() < 1e-11, "1D ll {ll1} vs {seq_ll}");

    let k_len = active_ts.len();
    let mut ms = mf.clone();
    let mut ps = pf.clone();
    for idx in (0..k_len).rev() {
        let g = pf[idx] / pp[idx];
        ms[idx] = mf[idx] + g * (ms[idx + 1] - mp[idx]);
        ps[idx] = pf[idx] + g * g * (ps[idx + 1] - pp[idx]);
    }
    // Check boundary t=0 (prior smoothed by all future data!) and every active timestamp
    let (m_prior_s, p_prior_s) = smooth_1d(&t1, 0, m0, p0).unwrap();
    assert!((m_prior_s - ms[0]).abs() < 1e-11);
    assert!((p_prior_s - ps[0]).abs() < 1e-11 && p_prior_s <= p0 + 1e-12);
    for (idx, &ts) in active_ts.iter().enumerate() {
        let (m_got, p_got) = smooth_1d(&t1, ts, m0, p0).unwrap();
        assert!((m_got - ms[idx + 1]).abs() < 1e-11);
        assert!((p_got - ps[idx + 1]).abs() < 1e-11);
        // Information inequality: smoothed variance <= filtered variance!
        assert!(p_got <= pf[idx + 1] + 1e-11);
    }

    // 2D fast-path vs general KalmanMonoid parity on RTS smoothing & log_marginal_likelihood
    let mut t2: FoldTree<KalmanFilter2D, 4> = FoldTree::new(());
    let mut tn: FoldTree<KalmanMonoid, 4> = FoldTree::new(2);
    for i in 1..=30i64 {
        let dt = rng.gen_range(0.3..1.7);
        let y = (i as f64) * 0.2 + rand_normal(&mut rng);
        t2.upsert(
            EventKey::at(i),
            KalmanFilter2D::from_constant_velocity(dt, 0.04, 0.3, y).unwrap(),
        );
        tn.upsert(
            EventKey::at(i),
            KalmanMonoid::from_constant_velocity(dt, 0.04, 0.3, y).unwrap(),
        );
    }
    let m0_2 = [0.1, -0.2];
    let p0_2 = [4.0, 0.1, 0.1, 2.0];
    let ll2 = t2.summary().log_marginal_likelihood(m0_2, p0_2).unwrap();
    let lln = tn.summary().log_marginal_likelihood(&m0_2, &p0_2).unwrap();
    assert!((ll2 - lln).abs() < 1e-11);
    for q_t in [0i64, 1, 15, 30, i64::MAX] {
        let (m2, p2) = smooth_2d(&t2, q_t, m0_2, p0_2).unwrap();
        let postn = smooth_nd(&tn, q_t, &m0_2, &p0_2).unwrap();
        for i in 0..2 {
            assert!((m2[i] - postn.mean[i]).abs() < 1e-11);
        }
        for i in 0..4 {
            assert!((p2[i] - postn.cov[i]).abs() < 1e-11);
        }
        // Smoothed diagonal variance <= filtered diagonal variance
        let (_, pf2) = t2.prefix(EventKey::time_end(q_t)).posterior(m0_2, p0_2).unwrap();
        assert!(p2[0] <= pf2[0] + 1e-11 && p2[3] <= pf2[3] + 1e-11);
    }

    // 3D vector-observation KalmanMonoid smooth_nd & log_marginal_likelihood finite & PSD
    let f3 = [1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.95];
    let q3 = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05];
    let h23 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let r22 = [0.2, 0.03, 0.03, 0.3];
    let mut t3: FoldTree<KalmanMonoid, 4> = FoldTree::new(3);
    for i in 1..=25i64 {
        let y = [(i as f64) * 0.1 + rand_normal(&mut rng) * 0.2, rand_normal(&mut rng) * 0.3];
        t3.upsert(
            EventKey::at(i),
            KalmanMonoid::from_vector_obs(3, 2, &f3, &q3, &h23, &r22, &y).unwrap(),
        );
    }
    let m0_3 = [0.0; 3];
    let p0_3 = [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0];
    let ll3 = t3.summary().log_marginal_likelihood(&m0_3, &p0_3).unwrap();
    assert!(ll3.is_finite() && ll3 < 0.0);
    let sm3 = smooth_nd(&t3, 12, &m0_3, &p0_3).unwrap();
    let fl3 = t3.prefix(EventKey::time_end(12)).posterior(&m0_3, &p0_3).unwrap();
    for d in 0..3 {
        assert!(sm3.cov[d * 3 + d] <= fl3.cov[d * 3 + d] + 1e-11);
    }

    // High-offset (1e6) Anchor-Centered Gaussian Potential test: zero O(y^2/S) log_z cancellation!
    let base = 1e6_f64;
    let mut t_hi: FoldTree<KalmanFilter2D, 4> = FoldTree::new(());
    let mut m_seq = [base, 0.0];
    let mut p_seq = [10.0, 0.0, 0.0, 10.0];
    let mut seq_ll_hi = 0.0;
    let q_acc = 1e-3;
    let r_obs = 0.25;
    let q_mat = [q_acc / 3.0, q_acc / 2.0, q_acc / 2.0, q_acc];
    for t in 0..2_000i64 {
        let y = base + 0.05 * (t as f64) + rand_normal(&mut rng) * 0.5;
        t_hi.upsert(
            EventKey::at(t),
            KalmanFilter2D::from_constant_velocity(1.0, q_acc, r_obs, y).unwrap(),
        );
        let mp0 = m_seq[0] + m_seq[1];
        let mp1 = m_seq[1];
        let pp00 = p_seq[0] + p_seq[1] + p_seq[2] + p_seq[3] + q_mat[0];
        let pp01 = p_seq[1] + p_seq[3] + q_mat[1];
        let pp11 = p_seq[3] + q_mat[3];
        let s = pp00 + r_obs;
        let innov = y - mp0;
        seq_ll_hi += -0.5 * (LN_TAU + s.ln() + innov * innov / s);
        let k0 = pp00 / s;
        let k1 = pp01 / s;
        m_seq = [mp0 + k0 * innov, mp1 + k1 * innov];
        let a00 = 1.0 - k0;
        let np00 = a00 * a00 * pp00 + k0 * k0 * r_obs;
        let np01 = a00 * (pp01 - k1 * pp00) + k0 * k1 * r_obs;
        let np11 = pp11 - 2.0 * k1 * pp01 + k1 * k1 * (pp00 + r_obs);
        p_seq = [np00, np01, np01, np11];
    }
    let bf_ll_hi = t_hi
        .summary()
        .log_marginal_likelihood([base, 0.0], [10.0, 0.0, 0.0, 10.0])
        .unwrap();
    let rel = (bf_ll_hi - seq_ll_hi).abs() / seq_ll_hi.abs().max(1.0);
    assert!(rel < 1e-11, "1e6 offset anchor-centered ll rel err {rel}");
}

// 9. CovarianceMonoid inline (dim=5) & heap (dim=10) random permutations, 1e12 offset exactness,
// 1e150 spike eviction with zero corruption, and batch Mahalanobis parity.
#[test]
fn covariance_monoid_inline_heap_1e12_and_spike_eviction() {
    let mut rng = StdRng::seed_from_u64(0xC0FA_B1E5);
    let base = 1e12_f64;

    for &dim in &[5usize, 10usize] {
        let mut tree: FoldTree<CovarianceMonoid, 4> = FoldTree::new(dim);
        let mut truth: BTreeMap<EventKey, Vec<f64>> = BTreeMap::new();

        for _ in 0..250 {
            let op = rng.gen_range(0..10);
            if op < 7 || truth.len() < dim + 2 {
                let ts = rng.gen_range(-100i64..100i64);
                let key = EventKey::at(ts);
                let vec: Vec<f64> = (0..dim).map(|_| base + rand_normal(&mut rng) * 2.0).collect();
                truth.insert(key, vec.clone());
                tree.upsert(key, CovarianceMonoid::singleton(ts, &vec).unwrap());
            } else {
                let keys: Vec<_> = truth.keys().copied().collect();
                let victim = keys[rng.gen_range(0..keys.len())];
                truth.remove(&victim);
                assert!(tree.retract(victim).is_some());
            }
        }

        // Insert a 1e150 catastrophic spike and then retract it -> ZERO residual corruption!
        let spike_key = EventKey::at(999_999);
        let spike_vec = vec![1e150; dim];
        tree.upsert(spike_key, CovarianceMonoid::singleton(999_999, &spike_vec).unwrap());
        assert!(tree.retract(spike_key).is_some());

        // Independent centered reference covariance on exact f64 residuals (v - base)
        let n = truth.len() as f64;
        let centered: Vec<Vec<f64>> = truth
            .values()
            .map(|v| v.iter().map(|&x| x - base).collect())
            .collect();
        let mut ref_mean = vec![0.0; dim];
        for row in &centered {
            for j in 0..dim {
                ref_mean[j] += row[j] / n;
            }
        }
        let mut ref_cov = vec![0.0; dim * dim];
        for row in &centered {
            for i in 0..dim {
                for j in 0..dim {
                    ref_cov[i * dim + j] += (row[i] - ref_mean[i]) * (row[j] - ref_mean[j]) / (n - 1.0);
                }
            }
        }
        let s = tree.summary();
        let got_cov = s.covariance(1);
        for i in 0..(dim * dim) {
            assert!(
                (got_cov[i] - ref_cov[i]).abs() < 1e-10,
                "dim={dim} cov[{i}] mismatch: {} vs {}",
                got_cov[i],
                ref_cov[i]
            );
        }

        // Batch vs single Mahalanobis exact parity
        let mut queries_flat = Vec::new();
        let mut queries = Vec::new();
        for _ in 0..10 {
            let q: Vec<f64> = (0..dim).map(|_| base + rand_normal(&mut rng) * 3.0).collect();
            queries_flat.extend_from_slice(&q);
            queries.push(q);
        }
        let batch_d2 = s.mahalanobis_sq_batch(&queries_flat, 1e-5).unwrap();
        for (q, &bd2) in queries.iter().zip(batch_d2.iter()) {
            let sd2 = s.mahalanobis_sq(q, 1e-5).unwrap();
            assert!((sd2 - bd2).abs() < 1e-12 && sd2 >= 0.0 && sd2.is_finite());
        }
    }

    // NaN / Inf rejection
    assert!(matches!(
        CovarianceMonoid::singleton(0, &[1.0, f64::NAN]),
        Err(BackfoldError::NonFiniteInput(_))
    ));
}
