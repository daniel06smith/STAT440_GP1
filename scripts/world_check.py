"""Which leak world is consistent with everything the public leaderboard told us?

    python scripts/world_check.py          # -> data/processed/worlds/leak_risk_<world>.csv, data/processed/world_check.csv

Eight public scores come from ONE public future: the uploaded leak-aware plan ($74.83M) and seven other uploads,
including four probes built to isolate one thing each. Every candidate world below predicts all eight on the same
simulated futures (paired, heavy-tailed costs, corrected formula). A world is kept only if it reproduces them all.
The worlds are a handful of interpretable hypotheses (batch-level spike sizes, background level, leak cost level),
not a fitted model with free knobs -- choosing among them uses aggregate facts shared by the public and private
futures, so it does not overfit the public leaderboard.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BASE = "batchwave_corrected_leakaware"
PUBLIC = {BASE: 74.83, "batchwave": 70.42, "batchwave_corrected": 72.92, "batchwave_s035_corrected_leakaware": 73.94,
          "probe_area": 68.73, "probe_oldpoly": 69.38, "probe_swap": 55.35, "probe_length": 68.84}
YEARS = list(range(2027, 2052))

# world -> dict(spike scale per batch selector, background scale, cost scale)
#   spike selectors: (lay year lo, lay year hi, material or None) -> multiplier on the spike hazard
WORLDS = {
    "batchwave": {},
    "spike6465_x0.35": {"spike": [(1964, 1966, None, 0.35)]},
    "w1965_off": {"spike": [(1965, 1966, "wrought iron", 0.05)]},
    "w1965_off_bg_x0.3": {"spike": [(1965, 1966, "wrought iron", 0.05)], "bg": 0.3},
    "w1965_off_bg_x0.1": {"spike": [(1965, 1966, "wrought iron", 0.05)], "bg": 0.1},
    "bg_x0.3": {"bg": 0.3},
    "bg_x0.1": {"bg": 0.1},
    "all_x0.3": {"spike_all": 0.3, "bg": 0.3},
    "cost_x0.5": {"cost": 0.5},
    "w1965_off_cost_x0.5": {"spike": [(1965, 1966, "wrought iron", 0.05)], "cost": 0.5},
    "w1965_off_bg_x0.3_cost_x0.7": {"spike": [(1965, 1966, "wrought iron", 0.05)], "bg": 0.3, "cost": 0.7},
}


def build_risk(name: str, spec: dict) -> Path:
    import batch_wave_hazard as bw
    import leak_calibration as lc
    out = ROOT / "data/processed/worlds" / f"leak_risk_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = bw.load(ROOT)
    f = bw.fit(d, lc.LAST_YEAR)
    at = np.flatnonzero(~(d["leak_year"] <= lc.LAST_YEAR))
    lay, mat = d["lay_dec"][at], d["mat"][at]
    on = f.spike_mask(d, at)
    hb = f.bg_rate_forecast(d, at) * spec.get("bg", 1.0)
    sc = np.full(len(at), spec.get("spike_all", 1.0))
    for lo, hi, m, x in spec.get("spike", []):
        sel = (lay >= lo) & (lay < hi) & ((mat == m) if m else True)
        sc[sel] = x
    base = pd.read_csv(ROOT / "data/processed/leak_risk_batchwave.csv")
    assert (base["Pipe ID"].to_numpy() == d["pid"][at]).all()
    r = base.copy()
    P = []
    for Y in YEARS:
        ds = bw.spike_log_surv(f.spike, lay, Y - lay) - bw.spike_log_surv(f.spike, lay, Y + 1.0 - lay)
        P.append(-np.expm1(-(hb + np.where(on, np.maximum(ds, 0.0), 0.0) * sc)))
    P = np.column_stack(P)
    for k, Y in enumerate(YEARS):
        r[f"p_{Y}"] = P[:, k]
    r["p_leak_annual"] = 1 - (np.prod(1 - P[:, :10], axis=1)) ** 0.1
    r["cost_if_leak"] = r["cost_if_leak"] * spec.get("cost", 1.0)
    r.to_csv(out, index=False)
    return out


def score_world(item):
    name, spec = item
    from pipeplan import simulator
    from pipeplan.pipeline import read_submission
    from pipeplan.simulator import empirical_cost_ratios, monte_carlo, prepare
    from pipeplan.world import load_world, read_risk_csv
    path = build_risk(name, spec)
    simulator.set_cost_formula("corrected")
    w = load_world(risk=read_risk_csv(path, yearly=True))
    ratios = empirical_cost_ratios(ROOT)
    draws = {}
    for p in PUBLIC:
        mc = monte_carlo(w, prepare(w, read_submission(ROOT / "submission" / f"submission_{p}.csv")),
                         n_draws=120, seed=51, cost_ratios=ratios, keep_draws=True)
        draws[p] = mc["score_draws"] / 1e6
    return name, draws


def main() -> None:
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = dict(ex.map(score_world, WORLDS.items()))
    obs = np.array([PUBLIC[BASE]] + [PUBLIC[p] - PUBLIC[BASE] for p in PUBLIC if p != BASE])
    labels = ["level " + BASE] + [f"d {p}" for p in PUBLIC if p != BASE]
    rows = []
    for name, dr in results.items():
        X = np.column_stack([dr[BASE]] + [dr[p] - dr[BASE] for p in PUBLIC if p != BASE])  # (draws, 8)
        mu, sd = X.mean(0), X.std(0, ddof=1)
        z = (obs - mu) / np.maximum(sd, 0.05)
        C = np.cov(X.T) + np.eye(X.shape[1]) * 0.05 ** 2
        m2 = float((obs - mu) @ np.linalg.solve(C, obs - mu))  # joint (Mahalanobis^2), accounts for correlation
        rows.append({"world": name, "mahalanobis2": m2, "max_abs_z": float(np.abs(z).max()),
                     **{f"pred[{l}]": round(float(m), 2) for l, m in zip(labels, mu)},
                     **{f"z[{l}]": round(float(v), 2) for l, v in zip(labels, z)}})
    df = pd.DataFrame(rows).sort_values("mahalanobis2")
    df.to_csv(ROOT / "data/processed/world_check.csv", index=False)
    pd.set_option("display.width", 250)
    print("observed:", dict(zip(labels, obs.round(2))))
    show = ["world", "mahalanobis2", "max_abs_z"] + [c for c in df.columns if c.startswith("z[")]
    print(df[show].round(2).to_string(index=False))
    print("\npredicted values:")
    print(df[["world"] + [c for c in df.columns if c.startswith("pred[")]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
