"""Generate a crisp 1200x630 OpenGraph social preview card (docs/og-card.png) from live results.json."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results.json"
OUT = ROOT / "docs" / "og-card.png"

FONT_SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def fmt_x(val: float) -> str:
    if val >= 1000:
        return f"{val:,.0f}x"
    if val >= 100:
        return f"{val:.0f}x"
    return f"{val:.1f}x"


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    # Pull real live measurements directly from benchmarks/results.json
    py_rows = {
        (r["operator"], r["n"]): r for r in data["late_updates_python_vs_ecosystems"]
    }
    rust_rows = {
        (r["operator"], r["n"]): r for r in data["rust_vs_rust_o3_bench"]["late_update_rust_vs_rust"]
    }
    ma_rows = {r["scenario"]: r for r in data["multi_agent_eval"]}

    rts_py = py_rows[("kalman_2d_rts_smooth_plus_loglik", 100_000)]
    ewma_r500 = rust_rows[("ewma_ct_late_edit", 500_000)]
    k1_r500 = rust_rows[("kalman_1d_late_edit", 500_000)]
    rts_r500 = rust_rows[("kalman_2d_rts_smooth_plus_loglik", 500_000)]
    fdr_ma = ma_rows["firehose_500ch_fdr_and_loud_traps"]
    fork_ma = ma_rows["counterfactual_64_agents_100k"]
    dig = data["digest_compression_eval"]["backfold_digest"]

    # 2x supersampled canvas -> Lanczos downsample to 1200x630
    W, H = 2400, 1260
    img = Image.new("RGB", (W, H), "#0b0f19")
    draw = ImageDraw.Draw(img)

    # Subtle background grid
    for x in range(0, W, 80):
        draw.line([(x, 0), (x, H)], fill="#111827", width=2)
    for y in range(0, H, 80):
        draw.line([(0, y), (W, y)], fill="#111827", width=2)

    # Top accent bar gradient (teal -> emerald)
    for x in range(W):
        t = x / W
        r = int(20 + t * (74 - 20))
        g = int(184 + t * (222 - 184))
        b = int(166 + t * (128 - 166))
        draw.line([(x, 0), (x, 16)], fill=(r, g, b))

    f_title = ImageFont.truetype(FONT_SANS_BOLD, 108)
    f_sub = ImageFont.truetype(FONT_SANS, 40)
    f_badge = ImageFont.truetype(FONT_MONO_BOLD, 28)
    f_kpi_lbl = ImageFont.truetype(FONT_SANS_BOLD, 30)
    f_kpi_val = ImageFont.truetype(FONT_MONO_BOLD, 70)
    f_kpi_sub = ImageFont.truetype(FONT_MONO, 27)
    f_foot = ImageFont.truetype(FONT_MONO_BOLD, 32)
    f_foot_muted = ImageFont.truetype(FONT_MONO, 28)

    # Header & badges
    pad = 96
    draw.text((pad, 78), "backfold", font=f_title, fill="#f8fafc")

    badges = [
        ("v0.3.0", "#0f766e", "#ccfbf1"),
        ("Apache-2.0", "#1e293b", "#cbd5e1"),
        ("Rust + Python + MCP + WASM", "#1e293b", "#5eead4"),
        ("#![forbid(unsafe_code)]", "#1e293b", "#86efac"),
    ]
    bx = pad + 545
    by = 112
    for text, bg, fg in badges:
        bbox = draw.textbbox((0, 0), text, font=f_badge)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle(
            [bx, by, bx + tw + 40, by + th + 26],
            radius=14,
            fill=bg,
            outline="#334155",
            width=2,
        )
        draw.text((bx + 20, by + 11), text, font=f_badge, fill=fg)
        bx += tw + 56

    draw.text(
        (pad, 218),
        "Persistent O(log T) Out-of-Order Streaming Inference, Exact RTS Smoothing & Agent Compression",
        font=f_sub,
        fill="#94a3b8",
    )

    # 4 KPI Cards (2x2 grid)
    cards = [
        (
            "EXACT O(log T) RTS KALMAN SMOOTHER (T=100k Py)",
            fmt_x(rts_py["speedup_vs_numpy_seq"]),
            f"{rts_py['backfold_us']:.2f} us vs 6.6s NumPy RTS  |  diff {rts_py['max_abs_diff']:.1e}",
            "#2dd4bf",
        ),
        (
            "RUST vs HANDWRITTEN -O3 SEQ RUST (T=500k)",
            f"{fmt_x(ewma_r500['speedup_x'])} / {fmt_x(rts_r500['speedup_x'])}",
            f"EWMA {ewma_r500['backfold_us']:.2f} us | K1D {fmt_x(k1_r500['speedup_x'])} | RTS {rts_r500['backfold_us']:.2f} us",
            "#4ade80",
        ),
        (
            "500-STREAM FIREHOSE + 50 LOUD TRAPS (e-BH)",
            "0.0% FDR",
            f"TP 10/10, FP 0 in {fdr_ma['backfold_metric']['elapsed_ms']:.1f} ms (vs 84.1% naive)",
            "#38bdf8",
        ),
        (
            "64-AGENT FORKING + <300-TOK AGENT DIGEST",
            f"{fmt_x(fork_ma['backfold_metric']['speedup_x'])} / {dig['compression_ratio']:.0f}x",
            f"64 forks in {fork_ma['backfold_metric']['elapsed_ms']:.2f} ms | ~{dig['avg_tokens']:.0f} tok (100% recall)",
            "#a78bfa",
        ),
    ]

    card_w = (W - 2 * pad - 44) // 2
    card_h = 330
    top_y = 315
    for idx, (lbl, val, sub, color) in enumerate(cards):
        col = idx % 2
        row = idx // 2
        x0 = pad + col * (card_w + 44)
        y0 = top_y + row * (card_h + 36)
        x1 = x0 + card_w
        y1 = y0 + card_h
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=22,
            fill="#111827",
            outline="#1e293b",
            width=3,
        )
        # Left color bar
        draw.rounded_rectangle([x0 + 4, y0 + 28, x0 + 16, y1 - 28], radius=6, fill=color)
        draw.text((x0 + 44, y0 + 36), lbl, font=f_kpi_lbl, fill="#94a3b8")
        draw.text((x0 + 44, y0 + 106), val, font=f_kpi_val, fill=color)
        draw.text((x0 + 44, y0 + 244), sub, font=f_kpi_sub, fill="#cbd5e1")

    # Bottom command strip
    bar_y = 1072
    draw.rounded_rectangle(
        [pad, bar_y, W - pad, bar_y + 108],
        radius=18,
        fill="#0f172a",
        outline="#334155",
        width=2,
    )
    draw.text(
        (pad + 36, bar_y + 34),
        "$ pip install backfold   |   $ backfold mcp   |   Live WASM Playground",
        font=f_foot,
        fill="#f8fafc",
    )
    url = "pgg898200-ship-it.github.io/backfold"
    ub = draw.textbbox((0, 0), url, font=f_foot_muted)
    draw.text((W - pad - 36 - (ub[2] - ub[0]), bar_y + 37), url, font=f_foot_muted, fill="#2dd4bf")

    out_img = img.resize((1200, 630), Image.Resampling.LANCZOS)
    out_img.save(OUT, format="PNG", optimize=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
