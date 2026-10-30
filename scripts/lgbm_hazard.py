"""Yearly LightGBM leak model, checked with the same rolling holdout as scripts/leak_calibration.py.

    python scripts/lgbm_hazard.py                    # the holdout report (LightGBM variants vs the hazard GLM)
    python scripts/lgbm_hazard.py --write-risk data/processed/leak_risk_lgbm.csv

The idea is the same as the hazard GLM: one row per pipe per year (a "pipe-year") while the pipe is still
unleaked, target = "leaked this year?". A gradient-boosted tree model replaces the regression, so it can find
non-smooth age shapes (the age 62-65 spike) and spatial patterns (leaks cluster geographically) by itself.

How it is set up
  * objective = poisson with log(exposure) as an offset (2019 counts as half a year). For rare events this is the
    same hazard as the cloglog GLM:  p = 1 - exp(-lambda).
  * Only information available at the cutoff year c is used; every holdout fit sees years <= c only.
  * "Neighbour" features: share of pipes within 250 m that leaked up to the previous year, and exactly in the
    previous year. For forecast years beyond c+1 they are frozen at their year-c values (we cannot know the future).
  * `+year` variants add calendar year. Trees cannot extrapolate: every forecast year is treated like the last year in
    the training data, i.e. "the recent level persists" (which is what the GLM's recalibration step also assumes).

Needs numpy / pandas / scipy / lightgbm.  Reads data/raw/{pipes,train}.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leak_calibration as lc  # noqa: E402

import lightgbm as lgb  # noqa: E402

NEIGHBOUR_RADIUS = 250.0  # metres
PARAMS = dict(objective="poisson", learning_rate=0.05, num_leaves=8, min_data_in_leaf=300,
              lambda_l2=10.0, feature_fraction=0.9, verbose=-1, seed=1, num_threads=2)
ROUNDS = 250
BASE = ["age", "mat", "surf", "loglen"]
SPATIAL = ["mx", "my"]
NBR = ["nb_cum", "nb_prev"]
RECAL_VARIANTS = ("age only", "base (no space)", "base+nbr")
VARIANTS = {
    "age only": ["age"],
    "base (no space)": BASE,
    "base+xy": BASE + SPATIAL,
    "base+nbr": BASE + NBR,
    "base+xy+nbr": BASE + SPATIAL + NBR,
    "base+nbr+year": BASE + NBR + ["year"],
}


# ----------------------------------------------------------------------------- data
def add_geometry(root: Path, d: dict) -> dict:
    """Attach mid-point coordinates and a pipe-neighbour matrix to the dict from lc.load (same pipe order)."""
    pipes = pd.read_csv(root / "data/raw/pipes.csv")
    pipes["Material"] = pipes["Material"].fillna("wrought iron")
    pipes = pipes[pipes["Material"] != "polyurethane"].reset_index(drop=True)
    assert (pipes["Pipe ID"].to_numpy() == d["pid"]).all(), "pipe order differs from leak_calibration.load"
    mx = ((pipes["GPS x1"] + pipes["GPS x2"]) / 2).to_numpy()
    my = ((pipes["GPS y1"] + pipes["GPS y2"]) / 2).to_numpy()
    tree = cKDTree(np.column_stack([mx, my]))
    pairs = tree.query_pairs(NEIGHBOUR_RADIUS, output_type="ndarray")
    n = len(mx)
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
    d2 = dict(d)
    d2.update(mx=mx, my=my, A=A, deg=deg,
              mat_code=np.array([lc.MATS.index(m) if m in lc.MATS else 0 for m in d["mat"]]),
              surf_code=np.array([lc.SURFS.index(s) for s in d["surf"]]))
    return d2


def neighbour_features(d: dict, idx: np.ndarray, year: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """nb_cum: share of neighbours that leaked by the end of year t-1; nb_prev: share that leaked exactly in t-1.
    t-1 is capped at the cutoff so nothing after the cutoff is ever used."""
    ly = np.where(np.isnan(d["leak_year"]), np.inf, d["leak_year"])
    cum, prev = {}, {}
    for y in np.unique(np.minimum(year - 1, cutoff)):
        cum[y] = (d["A"] @ (ly <= y).astype(float)) / d["deg"]
        prev[y] = (d["A"] @ (ly == y).astype(float)) / d["deg"]
    ycap = np.minimum(year - 1, cutoff)
    nb_cum, nb_prev = np.empty(len(idx)), np.empty(len(idx))
    for y in np.unique(ycap):
        m = ycap == y
        nb_cum[m], nb_prev[m] = cum[y][idx[m]], prev[y][idx[m]]
    return nb_cum, nb_prev


def feature_frame(d: dict, idx: np.ndarray, year: np.ndarray, cutoff: int) -> pd.DataFrame:
    nb_cum, nb_prev = neighbour_features(d, idx, year, cutoff)
    return pd.DataFrame({
        "age": year - d["lay"][idx], "mat": d["mat_code"][idx], "surf": d["surf_code"][idx],
        "loglen": d["loglen"][idx], "mx": d["mx"][idx], "my": d["my"][idx],
        "nb_cum": nb_cum, "nb_prev": nb_prev, "year": year,
    })


# ----------------------------------------------------------------------------- model
def fit_lgbm(d: dict, c: int, cols: list[str]):
    py = lc.person_years(d, c)
    X = feature_frame(d, py["idx"], py["year"], c)[cols]
    cats = [k for k in ("mat", "surf") if k in cols]
    ds = lgb.Dataset(X, label=py["event"], init_score=np.log(py["expo"]), categorical_feature=cats, free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS), cols


def lgbm_p(fitted, d: dict, risk_idx: np.ndarray, year: int, c: int) -> np.ndarray:
    """Yearly leak probability for the pipes at positions `risk_idx` of d, in calendar `year`."""
    model, cols = fitted
    yr = np.full(len(risk_idx), float(year))
    # trees treat any year beyond the training data like the last training year; make that explicit
    X = feature_frame(d, risk_idx, np.where(yr > c, yr, yr), c)[cols].copy()
    if "year" in cols:
        X["year"] = np.minimum(X["year"], c)
    lam = np.exp(np.minimum(model.predict(X, raw_score=True), 20.0))
    return -np.expm1(-lam)


def oos_level_shift(d: dict, c: int, cols: list[str], k: int = 2) -> float:
    """Multiplier on the hazard so that predicted = observed leaks in the last k years, measured OUT OF SAMPLE:
    the model fitted on years <= y-1 predicts year y, for y = c-k+1..c. (The GLM's recalibration compares
    in-sample numbers, which does not work for a flexible tree model: it fits its own training years.)"""
    obs = exp = 0.0
    for y in range(c - k + 1, c + 1):
        fitted = fit_lgbm(d, y - 1, cols)
        at = np.flatnonzero(~(d["leak_year"] <= y - 1))
        exp += lgbm_p(fitted, d, at, y, y - 1).sum()
        obs += float((d["leak_year"][at] == y).sum())
    return float(np.clip(obs / exp, 0.5, 4.0))


def apply_shift(p: np.ndarray, m: float) -> np.ndarray:
    """Scale the hazard by m:  1-(1-p)^m."""
    return 1.0 - (1.0 - p) ** m


# ----------------------------------------------------------------------------- evaluation
def run_holdout(d: dict, cutoffs=(2021, 2022, 2023, 2024, 2025), horizons=(1, 3), variants=None) -> pd.DataFrame:
    variants = variants or list(VARIANTS)
    rows = []
    for c in cutoffs:
        at = np.flatnonzero(~(d["leak_year"] <= c))
        risk = lc.sub(d, ~(d["leak_year"] <= c))
        chat = lc.cost_hat(lc.fit_cost(d, c), risk)
        haz = lc.recalibrate(lc.fit_hazard(d, c), d)
        fits = {name: fit_lgbm(d, c, VARIANTS[name]) for name in variants}
        shifts = {name: oos_level_shift(d, c, VARIANTS[name]) for name in variants if name in RECAL_VARIANTS}
        for h in horizons:
            if c + h > lc.LAST_YEAR:
                continue
            ly = risk["leak_year"]
            actual = (ly > c) & (ly <= c + h)
            act_n, act_usd = int(actual.sum()), float(risk["cost"][actual].sum())
            models = {"hazard GLM + recal": np.column_stack([lc.hazard_p(haz, risk, c + 1 + k) for k in range(h)])}
            for name in variants:
                models["lgbm: " + name] = np.column_stack([lgbm_p(fits[name], d, at, c + 1 + k, c) for k in range(h)])
            for name in shifts:
                models["lgbm: " + name + " +recal"] = apply_shift(models["lgbm: " + name], shifts[name])
            for name, P in models.items():
                pcum = 1 - np.prod(1 - P, axis=1)
                el = pcum * chat
                k = max(1, risk["n"] // 10)
                top_usd, top_n = np.argsort(-el)[:k], np.argsort(-pcum)[:k]
                p1 = np.clip(P[:, 0], 1e-9, 1 - 1e-9)
                y1 = (ly == c + 1).astype(float)
                rows.append({
                    "cutoff": c, "horizon": h, "model": name,
                    "pred_leaks": pcum.sum(), "actual_leaks": act_n,
                    "top10_share_$": float(risk["cost"][actual & np.isin(np.arange(risk["n"]), top_usd)].sum() / max(act_usd, 1)),
                    "top10_share_leaks": float((actual & np.isin(np.arange(risk["n"]), top_n)).sum() / max(act_n, 1)),
                    "logloss_1y": float(-(y1 * np.log(p1) + (1 - y1) * np.log(1 - p1)).mean()),
                })
    return pd.DataFrame(rows)


def write_risk(d: dict, out: Path, variant: str, level_scale: float = 0.5) -> None:
    """Fit on ALL data (2019-2026), write yearly probabilities for 2027-2051 for the pipes still non-poly in 2027.
    p_leak_annual = 10-year-average yearly probability, usable by build_plan.py with --horizon-years 1."""
    c = lc.LAST_YEAR
    fitted = fit_lgbm(d, c, VARIANTS[variant])
    at = np.flatnonzero(~(d["leak_year"] <= c))
    risk = lc.sub(d, ~(d["leak_year"] <= c))
    years = list(range(c + 1, c + 26))
    P = np.column_stack([lgbm_p(fitted, d, at, y, c) for y in years])
    m = oos_level_shift(d, c, VARIANTS[variant])
    # The plan backtest showed the level barely matters (full, half or no correction scored within 0.4 points), while
    # the full correction can be extreme (x2.6 for 2026), so by default apply only half of it on the log scale.
    P = apply_shift(P, m ** level_scale)
    print(f"out-of-sample level multiplier: {m:.2f}  (applied: {m ** level_scale:.2f}, level_scale={level_scale})")
    p10 = 1 - np.prod(1 - P[:, :10], axis=1)
    df = pd.DataFrame(P, columns=[f"p_{y}" for y in years])
    df.insert(0, "p_leak_annual", 1 - (1 - p10) ** 0.1)
    df.insert(0, "cost_if_leak", lc.cost_hat(lc.fit_cost(d, c), risk))
    df.insert(0, "Pipe ID", risk["pid"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} pipes; mean yearly p 2027={P[:, 0].mean():.3f}, 2035={P[:, 8].mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--write-risk", default=None)
    ap.add_argument("--variant", default="base (no space)", choices=list(VARIANTS))
    ap.add_argument("--level-scale", type=float, default=0.5,
                    help="0 = no level correction, 1 = full out-of-sample correction, default 0.5 (half, log scale)")
    args = ap.parse_args()
    root = Path(args.root)
    d = add_geometry(root, lc.load(root))
    if args.write_risk:
        write_risk(d, Path(args.write_risk) if Path(args.write_risk).is_absolute() else root / args.write_risk, args.variant, args.level_scale)
        return
    res = run_holdout(d)
    res["leaks_ratio"] = res["pred_leaks"] / res["actual_leaks"]
    pd.set_option("display.width", 220)
    for h in sorted(res["horizon"].unique()):
        t = res[res["horizon"] == h]
        s = t.groupby("model").agg(level_ratio=("leaks_ratio", "mean"), top10_share_usd=("top10_share_$", "mean"),
                                   top10_share_leaks=("top10_share_leaks", "mean"), logloss_1y=("logloss_1y", "mean"))
        print(f"\n=== next {h} year(s), average over cutoffs (level_ratio 1.00 = right number of leaks) ===")
        print(s.round(3).sort_values("top10_share_leaks", ascending=False).to_string())
    print("\n--- by cutoff, horizon 1: top-10% share of actual leaks ---")
    print(res[res.horizon == 1].pivot(index="model", columns="cutoff", values="top10_share_leaks").round(3).to_string())


if __name__ == "__main__":
    main()
