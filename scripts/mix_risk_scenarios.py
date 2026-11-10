"""Mix two leak-risk scenarios into one risk file: "with probability q, world B is the true one".

    python scripts/mix_risk_scenarios.py --a data/processed/leak_risk_batchwave.csv \
        --b data/processed/leak_risk_batchwave_extrap.csv --q 0.5 --output data/processed/leak_risk_batchwave_mix50.csv

Why: the 1971 and 1982 cast-iron batches (~2,600 unleaked pipes) leak at a flat ~5%/y since 2022 and show no
age gradient yet, but the 1964-65 cast-iron batches showed the same flat plateau right before their age spike.
Whether the 1966-1985 batches spike (batchwave_extrap) or not (batchwave) cannot be told from 2019-2026, so
plan for the mixture instead of betting on one.

The mixture is taken on SURVIVAL curves, not on the yearly probabilities:
    S(y) = (1 - q) S_a(y) + q S_b(y),   p_y = 1 - S(y + 1) / S(y)
(averaging yearly probabilities would be wrong: in the spike world, pipes that survive the spike are
the non-defective ones). cost_if_leak must agree between the two files (same cost model); it is copied.

Output schema: same as every leak_risk_*.csv (Pipe ID, cost_if_leak, p_leak_annual, p_2027 .. p_2051).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def mix(a: pd.DataFrame, b: pd.DataFrame, q: float) -> pd.DataFrame:
    a = a.set_index("Pipe ID")
    b = b.set_index("Pipe ID").reindex(a.index)
    assert b.notna().all().all(), "the two risk files must cover the same pipes"
    assert np.allclose(a["cost_if_leak"], b["cost_if_leak"]), "cost_if_leak differs between the files"
    cols = sorted(c for c in a.columns if c.startswith("p_20"))
    Sa = np.cumprod(np.c_[np.ones(len(a)), 1 - a[cols].to_numpy()], axis=1)  # S(start of each year)
    Sb = np.cumprod(np.c_[np.ones(len(b)), 1 - b[cols].to_numpy()], axis=1)
    S = (1 - q) * Sa + q * Sb
    P = 1 - S[:, 1:] / np.maximum(S[:, :-1], 1e-300)
    out = pd.DataFrame(P, index=a.index, columns=cols).clip(0, 1)
    p10 = 1 - S[:, 10]  # P(leak within the first 10 years)
    out.insert(0, "p_leak_annual", 1 - (1 - p10) ** 0.1)
    out.insert(0, "cost_if_leak", a["cost_if_leak"])
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="world A (weight 1 - q)")
    ap.add_argument("--b", type=Path, required=True, help="world B (weight q)")
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not 0 <= args.q <= 1:
        ap.error("--q must be in [0, 1]")
    out = mix(pd.read_csv(args.a), pd.read_csv(args.b), args.q)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} pipes, q = {args.q})")


if __name__ == "__main__":
    main()
