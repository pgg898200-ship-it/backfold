"""Out-of-Order Telemetry + Bitemporal Audit + Multi-Stream e-BH FDR + Agent Digest.

Demonstrates an end-to-end streaming workflow combining:
1. Out-of-order late updates & O(log T) exact first-alarm tracking (`CusumStream`).
2. Zero-lookahead bitemporal audit (`BitemporalMoments.as_of(knowledge_time)`).
3. Multi-channel anytime-valid FDR control under loud distractors (`EProcessStream` + `ebh_reject`).
4. Lossless-recoverable <350-token LLM stream compression (`digest_stream`).
"""

from __future__ import annotations

import backfold as bf
import numpy as np


def main() -> None:
    rng = np.random.default_rng(123)

    # 1. Retroactive CUSUM alarm update when delayed packets arrive
    n = 20_000
    ts = np.arange(n, dtype=np.int64)
    zs = rng.normal(0.0, 1.0, size=n)
    zs[12_000:12_030] += 2.5

    cs = bf.CusumStream(mu0=0.0, sigma0=1.0, slack_k=0.5)
    cs.ingest(ts, zs)
    alarm_before = cs.first_alarm(10.0)
    # 8 delayed anomalous ticks arrive for t = 3,500..3,507
    for t in range(3_500, 3_508):
        cs.upsert(t, 3.2)
    alarm_after = cs.first_alarm(10.0)
    print(f"CUSUM first_alarm moved retroactively: {alarm_before} -> {alarm_after}")
    assert alarm_after is not None and alarm_before is not None and alarm_after < alarm_before

    # 2. Zero-lookahead bitemporal audit
    bt = bf.BitemporalMoments()
    bt.ingest_event(event_time=100, knowledge_time=100, x=10.0)
    bt.ingest_event(event_time=200, knowledge_time=200, x=20.0)
    # Late restatement of event_time=100 arrives at knowledge_time=500
    bt.ingest_event(event_time=100, knowledge_time=500, x=90.0)
    print(
        f"Bitemporal mean: as_of(250)={bt.as_of(250)['mean']:.1f}, "
        f"as_of(500)={bt.as_of(500)['mean']:.1f} (zero lookahead leakage!)"
    )

    # 3. 100-channel anytime-valid e-BH FDR sifting (5 true shifts + 10 loud 3x-noise traps)
    e_values = []
    for ch in range(100):
        sigma = 3.0 if 5 <= ch < 15 else 1.0
        raw = rng.normal(0.0, sigma, size=300)
        if ch < 5:
            raw[150:] += 2.0 * sigma
        med, mad = bf.causal_robust_scale(raw[:100])
        ep = bf.EProcessStream()
        ep.ingest_z(np.arange(100, 300, dtype=np.int64), (raw[100:] - med) / mad, null_margin=0.15)
        e_values.append(ep.e_value())
    rejected = bf.ebh_reject(e_values, alpha=0.05)
    tp = sum(1 for c in rejected if c < 5)
    fp = len(rejected) - tp
    print(f"100-channel e-BH FDR (alpha=0.05): rejected={rejected} -> TP={tp}/5, FP={fp}")

    # 4. Compress 20k-tick stream into <350-token agent digest
    xs = rng.normal(0.0, 1.0, size=n)
    xs[8_000:8_005] += 9.0
    xs[14_000:] += 2.0
    dig = bf.digest_stream("telemetry_ch0", ts, xs, top_k=4, threshold=12.0)
    print(
        f"Agent digest: {dig['raw_approx_tokens']:,} raw tok -> {dig['digest_approx_tokens']} tok "
        f"({dig['compression_ratio']:.0f}x compression) | first_alarm={dig['first_alarm_ts']}"
    )


if __name__ == "__main__":
    main()
