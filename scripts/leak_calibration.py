"""Leak-model calibration check (rolling holdout) + optional risk-file writer.

Question this answers: if we fit a leak model on data up to year c, does it predict the
*level* (number and dollar cost of leaks) in the following year(s), not just the ranking?

    python scripts/leak_calibration.py                       # the holdout report
    python scripts/leak_calibration.py --write-risk data/processed/leak_risk_hazard_v2.csv

Needs only numpy / pandas / scipy (already in pyproject.toml). Reads data/raw/{pipes,train}.csv.

Models compared (all fitted ONLY on years <= cutoff):
  constant     pooled yearly leak rate, same for every pipe (the "level" baseline)
  cheryl-style cross-sectional logistic "leaked by cutoff?" -> yearly p = 1-(1-p)^(1/years)
               (polynomial in lay year instead of her spline; same idea and same conversion)
  hazard       person-year model (Kenson's idea): P(leak in year t | alive) with age, age^2,
               material x age, surface, length. Piecewise-constant hazard (cloglog link) with
               an exposure offset, so 2019 counts as a half year (leaks only start in July 2019).
  hazard+trend the same plus a linear calendar-year term
  hazard+recal the same as `hazard`, but the level is rescaled by observed/expected leaks over
               the last 2 training years (a simple fix for a level that has drifted up)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FIRST_YEAR, LAST_YEAR = 2019, 2026
MATS = ["wrought iron", "cast iron", "gray iron", "copper", "brass"]  # first = base level
SURFS = ["grassland", "farmland", "swamp", "water", "road", "structure"]  # first = base level
RECAL_YEARS = 2
RIDGE = 1.0


# ----------------------------------------------------------------------------- data
def load(root: Path) -> dict:
    pipes = pd.read_csv(root / "data/raw/pipes.csv", parse_dates=["Lay date"])
    leaks = pd.read_csv(root / "data/raw/train.csv", parse_dates=["Date"])
    pipes["Material"] = pipes["Material"].fillna("wrought iron")  # the 8 blank-material pipes
    pipes = pipes[pipes["Material"] != "polyurethane"].reset_index(drop=True)
    lay = pipes["Lay date"].dt.year.astype(float)
    lay = lay.fillna(lay[pipes["Material"] == "wrought iron"].median())  # the 15 blank lay dates
    length = np.hypot(pipes["GPS x2"] - pipes["GPS x1"], pipes["GPS y2"] - pipes["GPS y1"])
    first = leaks.assign(year=leaks["Date"].dt.year).set_index("Pipe ID")
    ly = pipes["Pipe ID"].map(first["year"]).to_numpy(dtype=float)
    cost = pipes["Pipe ID"].map(first["Cost"]).to_numpy(dtype=float)
    return {
        "pid": pipes["Pipe ID"].to_numpy(), "lay": lay.to_numpy(), "mat": pipes["Material"].to_numpy(),
        "surf": pipes["Surface"].to_numpy(), "loglen": np.log(np.maximum(length.to_numpy(), 0.5)),
        "leak_year": ly, "cost": cost, "n": len(pipes),
    }


def sub(d: dict, mask: np.ndarray) -> dict:
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()} | {"n": int(mask.sum())}


# ----------------------------------------------------------------------------- hazard model
def design(mat, surf, lay, loglen, year, age_max, trend=False):
    age = np.minimum(year - lay, age_max)  # age is held flat beyond what the training data saw
    a = (age - 60.0) / 20.0
    a2 = a * a
    ll = loglen - 2.5
    cols = [np.ones(len(a)), a, a2]
    for m in MATS[1:]:
        ind = (mat == m).astype(float)
        cols += [ind, ind * a, ind * a2]
    cols += [(surf == s).astype(float) for s in SURFS[1:]]
    cols += [ll, a * ll]
    if trend:
        cols.append((year - FIRST_YEAR) / 6.0)
    return np.column_stack(cols)


def person_years(d: dict, last_year: int) -> dict:
    """One row per pipe per year while it is still alive (not yet leaked) up to `last_year`."""
    ly = np.where(d["leak_year"] <= last_year, d["leak_year"], np.nan)
    idx, yr, ev, ex = [], [], [], []
    for y in range(FIRST_YEAR, last_year + 1):
        alive = np.isnan(ly) | (ly >= y)
        i = np.flatnonzero(alive)
        idx.append(i); yr.append(np.full(len(i), float(y)))
        ev.append((ly[i] == y).astype(float)); ex.append(np.full(len(i), 0.5 if y == FIRST_YEAR else 1.0))
    idx, yr, ev, ex = map(np.concatenate, (idx, yr, ev, ex))
    return {"idx": idx, "year": yr, "event": ev, "expo": ex}


def fit_hazard(d: dict, last_year: int, trend: bool = False) -> dict:
    py = person_years(d, last_year)
    i = py["idx"]
    age_max = float((py["year"] - d["lay"][i]).max())
    X = design(d["mat"][i], d["surf"][i], d["lay"][i], d["loglen"][i], py["year"], age_max, trend)
    y, off = py["event"], np.log(py["expo"])
    pen = np.full(X.shape[1], RIDGE); pen[0] = 0.0

    def nll(beta):
        eta = np.minimum(X @ beta + off, 20.0)
        mu = np.exp(eta)
        ll = -y * np.log(-np.expm1(-mu) + 1e-300) + (1 - y) * mu
        g_eta = (1 - y) * mu - y * mu / np.expm1(mu)
        return ll.sum() + 0.5 * (pen * beta**2).sum(), X.T @ g_eta + pen * beta

    b0 = np.zeros(X.shape[1]); b0[0] = np.log(max(y.sum() / py["expo"].sum(), 1e-6))
    res = minimize(nll, b0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "maxfun": 4000})
    model = {"beta": res.x, "age_max": age_max, "trend": trend, "shift": 0.0, "last_year": last_year}
    model["converged"] = bool(res.success)
    return model


def hazard_p(model: dict, d: dict, year: int, shift: float = 0.0) -> np.ndarray:
    X = design(d["mat"], d["surf"], d["lay"], d["loglen"], np.full(d["n"], float(year)),
               model["age_max"], model["trend"])
    eta = np.minimum(X @ model["beta"] + model["shift"] + shift, 20.0)
    return -np.expm1(-np.exp(eta))


def recalibrate(model: dict, d: dict, k: int = RECAL_YEARS) -> dict:
    """Shift the level so that expected = observed leaks over the last k training years."""
    last = model["last_year"]
    obs = exp = 0.0
    for y in range(last - k + 1, last + 1):
        alive = np.isnan(np.where(d["leak_year"] <= last, d["leak_year"], np.nan)) | (d["leak_year"] >= y)
        s = sub(d, alive)
        exp += hazard_p(model, s, y).sum()
        obs += float((s["leak_year"] == y).sum())
    return {**model, "shift": float(np.log(obs / exp))}


# ----------------------------------------------------------------------------- cost model
def cost_X(d):
    cols = [np.ones(d["n"])] + [(d["mat"] == m).astype(float) for m in MATS[1:]]
    cols += [(d["surf"] == s).astype(float) for s in SURFS[1:]] + [d["loglen"] - 2.5]
    return np.column_stack(cols)


def fit_cost(d: dict, last_year: int) -> np.ndarray:
    """Gamma regression with log link (same as Cheryl/Kenson): E[cost | leak]."""
    m = d["leak_year"] <= last_year
    X, y = cost_X(sub(d, m)), d["cost"][m]
    beta = np.zeros(X.shape[1]); beta[0] = np.log(y.mean())
    for _ in range(40):
        mu = np.exp(np.clip(X @ beta, -30, 30))
        z = X @ beta + (y - mu) / mu
        beta = np.linalg.solve(X.T @ X + 1e-6 * np.eye(len(beta)), X.T @ z)
    return beta


def cost_hat(beta: np.ndarray, d: dict) -> np.ndarray:
    return np.exp(np.clip(cost_X(d) @ beta, -30, 30))


# ----------------------------------------------------------------------------- Cheryl-style
def cheryl_X(d):
    t = (d["lay"] - 1965.0) / 10.0
    cols = [np.ones(d["n"]), t, t**2, t**3, t**4]
    cols += [(d["mat"] == m).astype(float) for m in MATS[1:]]
    cols += [(d["surf"] == s).astype(float) for s in SURFS[1:]] + [d["loglen"] - 2.5]
    return np.column_stack(cols)


def fit_cheryl(d: dict, last_year: int):
    X = cheryl_X(d)
    y = (d["leak_year"] <= last_year).astype(float)
    b = np.zeros(X.shape[1])
    for _ in range(30):  # IRLS for logistic regression
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-9)
        b = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (w * (X @ b) + (y - p)))
    return b


def cheryl_annual(b, d: dict, last_year: int) -> np.ndarray:
    p = 1 / (1 + np.exp(-np.clip(cheryl_X(d) @ b, -30, 30)))
    years = last_year - FIRST_YEAR + 1  # she used 8 for 2019-2026
    return 1 - (1 - p) ** (1.0 / years)


# ----------------------------------------------------------------------------- evaluation
def yearly_matrix(model_name: str, fitted, d_all: dict, risk: dict, c: int, h: int) -> np.ndarray:
    """(n_at_risk, h) matrix of yearly leak probabilities for years c+1 .. c+h."""
    cols = []
    for k in range(h):
        if model_name == "constant":
            cols.append(np.full(risk["n"], fitted))
        elif model_name == "cheryl-style":
            cols.append(cheryl_annual(fitted[0], risk, c))
        else:
            cols.append(hazard_p(fitted, risk, c + 1 + k))
    return np.column_stack(cols)


def run_holdout(d: dict, cutoffs=(2021, 2022, 2023, 2024, 2025), horizons=(1, 3)) -> pd.DataFrame:
    rows, last_detail = [], None
    for c in cutoffs:
        at_risk = ~(d["leak_year"] <= c)  # not leaked by the cutoff (leak_year NaN or later)
        risk = sub(d, at_risk)
        # fit everything on data <= c only
        py = person_years(d, c)
        const = py["event"].sum() / py["expo"].sum()
        haz = fit_hazard(d, c)
        models = {
            "constant": const,
            "cheryl-style": (fit_cheryl(d, c),),
            "hazard": haz,
            "hazard+trend": fit_hazard(d, c, trend=True),
            "hazard+recal": recalibrate(haz, d),
        }
        cb = fit_cost(d, c)
        chat = cost_hat(cb, risk)
        for h in horizons:
            if c + h > LAST_YEAR:
                continue
            ly = risk["leak_year"]
            actual = (ly > c) & (ly <= c + h)
            act_n, act_usd = int(actual.sum()), float(risk["cost"][actual].sum())
            for name, fitted in models.items():
                P = yearly_matrix(name, fitted, d, risk, c, h)
                pcum = 1 - np.prod(1 - P, axis=1)
                exp_loss = pcum * chat
                top = np.argsort(-exp_loss)[: max(1, risk["n"] // 10)]
                rows.append({
                    "cutoff": c, "horizon": h, "model": name,
                    "pred_leaks": pcum.sum(), "actual_leaks": act_n,
                    "pred_usd_M": exp_loss.sum() / 1e6, "actual_usd_M": act_usd / 1e6,
                    "top10pct_share_of_actual_usd": float(risk["cost"][actual & np.isin(np.arange(risk["n"]), top)].sum() / max(act_usd, 1)),
                })
                if c == 2025 and h == 1:
                    last_detail = (name, pcum, actual)
    return pd.DataFrame(rows), last_detail


def decile_table(d: dict, model_name: str = "hazard+recal", c: int = 2025) -> pd.DataFrame:
    risk = sub(d, ~(d["leak_year"] <= c))
    haz = fit_hazard(d, c)
    fitted = recalibrate(haz, d) if model_name == "hazard+recal" else haz
    p = hazard_p(fitted, risk, c + 1)
    actual = (risk["leak_year"] == c + 1).astype(float)
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": p, "actual": actual, "decile": q}).groupby("decile").agg(
        pipes=("pred", "size"), mean_pred=("pred", "mean"), actual_rate=("actual", "mean"))


def write_risk(d: dict, out: Path) -> None:
    """Fit on ALL data (2019-2026) and write yearly leak probabilities 2027-2051 for the pipes
    that are still non-polyurethane in 2027. `p_leak_annual` (10-year-average yearly probability)
    makes the file usable by the existing pipeline with --p-col p_leak_annual --horizon-years 1."""
    model = recalibrate(fit_hazard(d, LAST_YEAR), d)
    beta_c = fit_cost(d, LAST_YEAR)
    risk = sub(d, ~(d["leak_year"] <= LAST_YEAR))
    years = list(range(LAST_YEAR + 1, LAST_YEAR + 26))
    P = np.column_stack([hazard_p(model, risk, y) for y in years])
    p10 = 1 - np.prod(1 - P[:, :10], axis=1)
    df = pd.DataFrame(P, columns=[f"p_{y}" for y in years])
    df.insert(0, "p_leak_annual", 1 - (1 - p10) ** (1 / 10))
    df.insert(0, "cost_if_leak", cost_hat(beta_c, risk))
    df.insert(0, "Pipe ID", risk["pid"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} pipes; mean yearly p 2027={P[:,0].mean():.3f}, 2035={P[:,8].mean():.3f}, "
          f"2051={P[:,-1].mean():.3f}; level shift {model['shift']:+.3f})")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--write-risk", default=None, help="write the refitted yearly risk file here")
    args = ap.parse_args()
    d = load(Path(args.root))

    # actual yearly hazard among pipes still at risk (the number the models must reproduce)
    print("Observed leaks per year among pipes still at risk (2019 = half-year of data):")
    alive = d["n"]
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        k = int((d["leak_year"] == y).sum())
        e = 0.5 if y == FIRST_YEAR else 1.0
        print(f"  {y}: at risk {alive:6d}  leaks {k:5d}  yearly rate {k / (alive * e):.3f}")
        alive -= k

    res, _ = run_holdout(d)
    pd.set_option("display.width", 200)
    for h in sorted(res["horizon"].unique()):
        t = res[res["horizon"] == h].copy()
        t["leaks_ratio"] = t["pred_leaks"] / t["actual_leaks"]
        t["usd_ratio"] = t["pred_usd_M"] / t["actual_usd_M"]
        print(f"\n=== Predicting the next {h} year(s); ratio = predicted / actual (1.00 = perfectly calibrated) ===")
        print(t.round(2).to_string(index=False))
        s = t.groupby("model").agg(mean_leaks_ratio=("leaks_ratio", "mean"), mean_usd_ratio=("usd_ratio", "mean"),
                                   mean_top10_share=("top10pct_share_of_actual_usd", "mean")).round(2)
        print("\naverage over cutoffs:\n" + s.to_string())

    print("\n=== Reliability, hazard+recal fitted on 2019-2025, predicting 2026, by predicted-risk decile ===")
    print(decile_table(d).round(3).to_string())

    if args.write_risk:
        write_risk(d, Path(args.root) / args.write_risk if not Path(args.write_risk).is_absolute() else Path(args.write_risk))


if __name__ == "__main__":
    main()
