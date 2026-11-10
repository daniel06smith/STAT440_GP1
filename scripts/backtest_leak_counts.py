"""Do the risk models under-predict how many leaks happen? Rolling-origin backtest of leak COUNTS (and $).

    python scripts/backtest_leak_counts.py

Why: both real leaderboard scores (batchwave $70.42M, ensemble w050 $71.70M) sit below the 5th percentile of
the heavy-tailed Monte Carlo under every risk model, which points at too few predicted leaks (or too little $).

For each cutoff c in 2021..2025, every model is refitted on data <= c only, then predicts each later year
Y = c+1..2026. Two views:
  * one-step   : among pipes that actually survived to the start of Y, sum of P(leak in Y) vs leaks in Y
                 (pure calibration of the yearly level; what the planner's p_YYYY columns claim)
  * cumulative : over all pipes at risk at c, sum of P(leak in c+1..Y) vs leaks in c+1..Y
                 (the planner-relevant quantity: compounding errors included)
Both in counts and in $ (P x cost_if_leak from the current Gamma cost GLM vs actual cost).

Models: hazard (= leak_calibration hazard+recal), lgbm (base+nbr, out-of-sample logistic calibration, as in
ensemble_risk.choose_lgbm_weight), ens_w050 / ens_w060 (hazard-space blends of the two), batchwave
(spike depends on lay date only), batchwave_mat (+ per-material failure-age / share offsets) and
batchwave_batch (+ per-batch offsets, batch = lay year x material). The _extrap
variant is not run: its extra 1966-1985 spikes only start ~2027, after the last backtest year.

Outputs (data/processed/):
  backtest_leak_counts.csv        cutoff, model, year, h (=Y-c), view (one_step|cumulative), group, level,
                                  pred_n, actual_n, pred_usd, actual_usd, ratio_n (= pred_n/actual_n),
                                  ratio_usd. group/level = all/all, surface/<s>, material/<m>, lay/<bucket>.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import batch_wave_hazard as bw  # noqa: E402
import leak_calibration as lc  # noqa: E402
from ensemble_risk import apply_logistic_calibrator, blend_probability_matrix, fit_oos_lgbm_calibrator  # noqa: E402
from lgbm_hazard import VARIANTS, add_geometry, fit_lgbm, lgbm_p  # noqa: E402

CUTOFFS = (2021, 2022, 2023, 2024, 2025)
LAY_BINS = [-np.inf, 1954, 1956, 1960, 1964, 1966, 1976, np.inf]
LAY_LABELS = ["<1954", "1954-55", "1956-59", "1960-63", "1964-65", "1966-75", "1976+"]


def predict_all(d: dict, dbw: dict, c: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pipes at risk at c (indices into d) and, per model, an (n_at_risk, 2026-c) matrix of P(leak in Y | alive)."""
    at = np.flatnonzero(~(d["leak_year"] <= c))
    risk = lc.sub(d, ~(d["leak_year"] <= c))
    years = list(range(c + 1, lc.LAST_YEAR + 1))

    haz = lc.recalibrate(lc.fit_hazard(d, c), d)
    p_haz = np.column_stack([lc.hazard_p(haz, risk, Y) for Y in years])

    cols = VARIANTS["base+nbr"]
    lg = fit_lgbm(d, c, cols)
    cal = fit_oos_lgbm_calibrator(d, c, cols, 3)
    p_lgb = np.column_stack([apply_logistic_calibrator(lgbm_p(lg, d, at, Y, c), cal) for Y in years])

    out = {"hazard": p_haz, "lgbm": p_lgb,
           "ens_w050": blend_probability_matrix(p_lgb, p_haz, 0.5),
           "ens_w060": blend_probability_matrix(p_lgb, p_haz, 0.6)}
    bw.MATERIAL_SPIKE = False
    for name, mode in (("batchwave", "none"), ("batchwave_mat", "material"), ("batchwave_batch", "batch")):
        bw.OFFSET_MODE = mode
        f = bw.fit(dbw, c)
        out[name] = bw.yearly_p(f, dbw, at, years)
        if mode == "batch":
            print(f"  cutoff {c} drift slope {f.spike.b:+.3f}; batch offsets: {bw.batch_offsets(f, dbw, 4)}", flush=True)
    bw.OFFSET_MODE = "none"
    return at, out


def groups(d: dict, at: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    lay = pd.cut(d["lay"][at], LAY_BINS, labels=LAY_LABELS, right=False).astype(str)
    g = [("all", "all", np.ones(len(at), bool))]
    g += [("surface", s, d["surf"][at] == s) for s in lc.SURFS]
    g += [("material", m, d["mat"][at] == m) for m in lc.MATS]
    g += [("lay", b, lay == b) for b in LAY_LABELS]
    return [x for x in g if x[2].any()]


def main() -> None:
    d = add_geometry(ROOT, lc.load(ROOT))
    dbw = bw.load(ROOT)
    assert (dbw["pid"] == d["pid"]).all()
    rows = []
    for c in CUTOFFS:
        at, preds = predict_all(d, dbw, c)
        risk = lc.sub(d, ~(d["leak_year"] <= c))
        cost_hat = lc.cost_hat(lc.fit_cost(d, c), risk)
        ly = d["leak_year"][at]
        act_cost = np.nan_to_num(d["cost"][at])
        gs = groups(d, at)
        for name, P in preds.items():
            surv = np.ones(len(at))  # model's P(still alive at start of Y)
            for k in range(P.shape[1]):
                Y = c + 1 + k
                alive = ~(ly < Y)
                hit = ly == Y
                cum_hit = ly <= Y
                p_cum = 1 - surv * (1 - P[:, k])
                for grp, lvl, m in gs:
                    a = m & alive
                    rows.append(dict(cutoff=c, model=name, year=Y, h=k + 1, view="one_step", group=grp, level=lvl,
                                     pred_n=P[a, k].sum(), actual_n=float((hit & m).sum()),
                                     pred_usd=(P[a, k] * cost_hat[a]).sum(), actual_usd=act_cost[hit & m].sum()))
                    rows.append(dict(cutoff=c, model=name, year=Y, h=k + 1, view="cumulative", group=grp, level=lvl,
                                     pred_n=p_cum[m].sum(), actual_n=float((cum_hit & m).sum()),
                                     pred_usd=(p_cum[m] * cost_hat[m]).sum(), actual_usd=act_cost[cum_hit & m].sum()))
                surv = surv * (1 - P[:, k])
        print(f"cutoff {c} done", flush=True)

    res = pd.DataFrame(rows)
    res["ratio_n"] = res["pred_n"] / res["actual_n"].replace(0, np.nan)
    res["ratio_usd"] = res["pred_usd"] / res["actual_usd"].replace(0, np.nan)
    out = ROOT / "data/processed/backtest_leak_counts.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    a = res[(res.group == "all")]
    print("\n=== one-step, all pipes: predicted / actual leaks, by year (rows) and cutoff (cols) — model averaged ===")
    for view in ("one_step", "cumulative"):
        s = a[a.view == view]
        print(f"\n--- {view}: ratio pred/actual LEAKS, by model x horizon (pooled over cutoffs) ---")
        t = s.groupby(["model", "h"])[["pred_n", "actual_n"]].sum()
        print((t.pred_n / t.actual_n).unstack("h").round(2).to_string())
        print(f"--- {view}: ratio pred/actual $ ---")
        t = s.groupby(["model", "h"])[["pred_usd", "actual_usd"]].sum()
        print((t.pred_usd / t.actual_usd).unstack("h").round(2).to_string())
    s = a[(a.view == "one_step")]
    print("\n--- one-step pred/actual leaks by calendar year (pooled over cutoffs) ---")
    t = s.groupby(["model", "year"])[["pred_n", "actual_n"]].sum()
    print((t.pred_n / t.actual_n).unstack("year").round(2).to_string())
    s = res[(res.view == "one_step") & (res.group == "all") & res.model.str.startswith("batchwave")]
    print("\n--- batchwave variants: one-step pred/actual leaks by cutoff (rows) and year ---")
    t = s.groupby(["model", "cutoff", "year"])[["pred_n", "actual_n"]].sum()
    print((t.pred_n / t.actual_n).unstack("year").round(2).to_string())
    for grp in ("lay", "surface", "material"):
        s = res[(res.view == "one_step") & (res.group == grp)]
        t = s.groupby(["level", "model"])[["pred_n", "actual_n"]].sum()
        r = (t.pred_n / t.actual_n).unstack("model").round(2)
        r["actual_n"] = t.actual_n.groupby("level").first()
        print(f"\n--- one-step pred/actual leaks by {grp} (pooled over cutoffs and years) ---")
        print(r.to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
