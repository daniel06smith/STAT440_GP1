"""Batch-wave leak model: a defect "spike" at a drifting failure age + a smoothed background hazard.

    python scripts/batch_wave_hazard.py                       # rolling-origin holdout report
    python scripts/batch_wave_hazard.py --write-risk data/processed/leak_risk_batchwave.csv

What the data show (see the EDA notes in CLAUDE.md, "batch failure waves")
  * ~82% of all 2019-2026 leaks happen at ages 61-67. Each installation batch fails in a tight window
    (sd of failure age ~0.5 y), about 77% of a batch fails in it, and the survivors drop back to ~0.01/y.
  * The failure age drifts younger for newer batches (~65 for 1955, 63.5 for 1959, 62.5 for 1963).
    Within a batch the leak date tracks the exact lay date one-for-one, so it is an AGE effect.
  * Outside the spike there is a background hazard whose level jumped around 2022-23 (strongly for the
    1971 and 1982 batches, mildly for the oldest pipes).

Model, for a pipe laid at (decimal) date L and observed from 2019.5 (left truncation) to its leak or the cutoff:
  hazard(t) = h_spike(age) + h_bg(t)
  h_spike   : with probability pi the pipe is "defective" and fails at age A ~ Normal(mu(L), sigma),
              mu(L) = a + b * (L - 1960). Survival S_s(age) = 1 - pi + pi * Q((age - mu) / sigma).
              Conditioning on survival to 2019.5 makes pipes whose spike is already over almost risk-free.
              Applied to pipes laid before SPIKE_LAST_LAY (default: the batches where a spike is observed);
              --extrapolate-spike also applies the fitted drift line to the 1966-1985 batches.
  h_bg      : log h_bg = material + surface + log length + gamma[group, calendar year], with a random-walk
              penalty on gamma over calendar years (a local-level time-series model per batch group).
              Forecast: gamma stays at its last smoothed level (random-walk forecast).
Fitted by EM over which component caused each leak. Only data up to the cutoff are used.

Output schema (same as the other leak_risk_*.csv files):
  Pipe ID, cost_if_leak ($, Gamma GLM from leak_calibration.py), p_leak_annual (10-year-average yearly
  probability, for the old constant-hazard planner), p_2027 .. p_2051 (P(leak in that year | not leaked yet)).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leak_calibration as lc  # noqa: E402

START = 2019.5  # leak recording effectively starts in July 2019 (12 leaks in Jan-Jun 2019)
SPIKE_LAST_LAY = 1966.0  # spike component applies to pipes laid before this date (unless extrapolating)
GROUP_EDGES = [1954.0, 1966.0, 1976.0]  # background groups: <1954 | 1954-65 | 1966-75 | 1976+
RW_PENALTY = 10.0  # random-walk smoothing of the yearly background level (per group)
RIDGE = 0.1
FAILURE_DIST = "normal"  # failure-age distribution of defective pipes: "normal" or "logistic" (holdout: identical)
# How the defective share pi depends on lay date. Observed: 1954-55 batches ~40%, 1957-62 batches a flat ~77%.
#   "constant": one pi (pulled down by 1954-55 -> under-predicts the 1963-65 waves by ~30%)
#   "linear"  : logit(pi) linear in lay date (fits the level, but overshoots to pi~0.86 for 1965, ~1 for 1982)
#   "step"    : one pi before PI_STEP_LAY, another after (matches the flat 77% plateau)
PI_MODE = "step"
PI_STEP_LAY = 1956.5
DRIFT_CURVATURE = False  # quadratic drift in lay date; holdout logloss 0.213 vs 0.171 linear (overfits), so off
# Batches of different materials laid in the same year fail at different ages and in different shares
# (1965 gray iron spiked in 2026 at ~61.7 while 1965 wrought iron has barely started; 1955 copper 58% vs
# gray 26% failed). MATERIAL_SPIKE adds a per-material shift to the failure age and to logit(pi)
# (wrought iron = 0), with a ridge penalty because some materials appear in only one or two batches.
# OFF by default: on the rolling holdout it fixes the gray-iron count (pred/actual 0.45 -> 1.03) but is
# slightly worse where it matters (logloss 0.1811 vs 0.1789, top-10% $ capture 0.772 vs 0.788) and its
# offsets swing between cutoffs (gray iron age shift -1.02y at 2021, -0.15y at 2025). --material-spike.
MATERIAL_SPIKE = False
N_MAT_OFFSETS = 4  # cast iron, gray iron, copper, brass (lc.MATS[1:])
SPIKE_RIDGE = 2.0
# BATCH offsets (OFFSET_MODE = "batch", --batch-spike): the deviations from the drift line belong to single
# installation batches, not to materials -- 1964 wrought iron fell right on the line (2026: 1,216 predicted,
# 1,208 actual) while 1965 wrought iron is late (103 predicted, 23 actual) and 1965 gray iron early (103 vs
# 362). Each batch (lay year x material, >= MIN_BATCH pipes) gets its own failure-age shift and logit(pi)
# shift, ridge-shrunk toward the drift line (prior sd ~ 1/sqrt(BATCH_RIDGE)); smaller batches share the line.
OFFSET_MODE = "none"  # "none" | "material" | "batch"   (MATERIAL_SPIKE = True is the same as "material")
MIN_BATCH = 150
BATCH_RIDGE = 2.0
N_OFFSETS = N_MAT_OFFSETS  # set by fit() from the mode


# ----------------------------------------------------------------------------- data
def load(root: Path) -> dict:
    """lc.load plus exact (decimal) lay and leak dates, in the same pipe order."""
    d = lc.load(root)
    pipes = pd.read_csv(root / "data/raw/pipes.csv", parse_dates=["Lay date"])
    pipes["Material"] = pipes["Material"].fillna("wrought iron")
    pipes = pipes[pipes["Material"] != "polyurethane"].reset_index(drop=True)
    assert (pipes["Pipe ID"].to_numpy() == d["pid"]).all()
    leaks = pd.read_csv(root / "data/raw/train.csv", parse_dates=["Date"]).set_index("Pipe ID")

    def dec(ts):
        return ts.dt.year + (ts.dt.dayofyear - 0.5) / 365.25

    lay = dec(pipes["Lay date"])
    lay = lay.fillna(lay[pipes["Material"] == "wrought iron"].median())
    d = dict(d)
    d["lay_dec"] = lay.to_numpy(float)
    d["leak_dec"] = dec(pipes["Pipe ID"].map(leaks["Date"])).to_numpy(float)
    d["mat_code"] = np.array([lc.MATS.index(m) for m in d["mat"]])
    # installation batch = lay year x material; big batches get codes 1..K (0 = shares the drift line)
    key = pd.Series(np.floor(d["lay_dec"]).astype(int).astype(str)) + " " + pd.Series(d["mat"])
    counts = key.value_counts()
    big = sorted(k for k, n in counts.items() if n >= MIN_BATCH and float(k.split()[0]) < SPIKE_LAST_LAY)
    d["batch_names"] = big
    d["batch_code"] = key.map({k: i + 1 for i, k in enumerate(big)}).fillna(0).astype(int).to_numpy()
    d["surf_code"] = np.array([lc.SURFS.index(s) for s in d["surf"]])
    d["group"] = np.searchsorted(GROUP_EDGES, d["lay_dec"], side="right")
    return d


# ----------------------------------------------------------------------------- model pieces
@dataclass
class Spike:
    a: float = 63.5  # failure age for a pipe laid in 1960
    b: float = -0.28  # change in failure age per year of lay date
    log_sigma: float = np.log(0.6)
    logit_pi: float = 1.1  # pi ~ 0.75
    c2: float = 0.0  # curvature of the drift: + c2 * (lay - 1960)^2
    pi_slope: float = 0.0  # PI_MODE "linear": change in logit(pi) per lay year; "step": offset before PI_STEP_LAY
    dmu: np.ndarray = field(default_factory=lambda: np.zeros(N_OFFSETS))  # failure-age shift by material/batch
    dpi: np.ndarray = field(default_factory=lambda: np.zeros(N_OFFSETS))  # logit(pi) shift by material/batch

    @property
    def sigma(self) -> float:
        return float(np.exp(self.log_sigma))

    @property
    def pi(self) -> float:
        """pi for a pipe laid in 1960."""
        return float(np.exp(self.log_pi(np.array([1960.0]))[0][0]))

    def pi_at(self, lay: float, mat: int = 0) -> float:
        return float(np.exp(self.log_pi(np.array([float(lay)]), np.array([mat]))[0][0]))

    def vec(self):
        return np.r_[self.a, self.b, self.log_sigma, self.logit_pi, self.c2, self.pi_slope, self.dmu, self.dpi]

    @staticmethod
    def _offset(off: np.ndarray, mat) -> np.ndarray:
        """off[mat - 1] for mat >= 1, 0 for wrought iron (mat 0) or when mat is None."""
        if mat is None:
            return 0.0
        return np.r_[0.0, off][np.asarray(mat, dtype=np.int64)]

    def log_pi(self, lay, mat=None):
        """log pi and log(1 - pi) for each lay date (and material code, lc.MATS order)."""
        if PI_MODE == "step":
            x = self.logit_pi + self.pi_slope * (lay < PI_STEP_LAY)
        else:
            x = self.logit_pi + self.pi_slope * (lay - 1960.0)
        x = x + self._offset(self.dpi, mat)
        return -np.logaddexp(0.0, -x), -np.logaddexp(0.0, x)

    def mu(self, lay, mat=None):
        x = lay - 1960.0
        return self.a + self.b * x + self.c2 * x * x + self._offset(self.dmu, mat)

    @staticmethod
    def of(v):
        v = np.asarray(v, dtype=float)
        k = (len(v) - 6) // 2
        return Spike(*map(float, v[:6]), dmu=v[6:6 + k].copy(), dpi=v[6 + k:6 + 2 * k].copy())


def _log_sf(z):
    return log_ndtr(-z) if FAILURE_DIST == "normal" else -np.logaddexp(0.0, z)


def _log_pdf(z):
    if FAILURE_DIST == "normal":
        return -0.5 * z * z - 0.5 * np.log(2 * np.pi)
    return -z - 2.0 * np.logaddexp(0.0, -z)


def spike_log_surv(sp: Spike, lay: np.ndarray, age: np.ndarray, mat=None) -> np.ndarray:
    """log S_s(age) = log(1 - pi + pi * Q(z)),  z = (age - mu(lay, mat)) / sigma."""
    z = (age - sp.mu(lay, mat)) / sp.sigma
    lp, l1p = sp.log_pi(lay, mat)
    return np.logaddexp(l1p, lp + _log_sf(z))


def spike_log_hazard(sp: Spike, lay: np.ndarray, age: np.ndarray, mat=None) -> np.ndarray:
    z = (age - sp.mu(lay, mat)) / sp.sigma
    log_f = sp.log_pi(lay, mat)[0] + _log_pdf(z) - np.log(sp.sigma)
    return log_f - spike_log_surv(sp, lay, age, mat)


def bg_design(d: dict, idx: np.ndarray) -> np.ndarray:
    cols = [(d["mat_code"][idx] == k).astype(float) for k in range(1, len(lc.MATS))]
    cols += [(d["surf_code"][idx] == k).astype(float) for k in range(1, len(lc.SURFS))]
    cols += [d["loglen"][idx] - 2.5]
    return np.column_stack(cols)


@dataclass
class Fit:
    spike: Spike
    beta: np.ndarray  # background covariate effects
    gamma: np.ndarray  # (n_groups, n_years) yearly log background level
    years: np.ndarray  # calendar years of gamma's columns
    cutoff: int
    extrapolate: bool
    loglik: float
    off_code: np.ndarray | None = None  # per pipe (all of d): which spike offset applies (0 = none)
    mode: str = "none"

    def spike_mask(self, d: dict, idx: np.ndarray) -> np.ndarray:
        return np.ones(len(idx), bool) if self.extrapolate else d["lay_dec"][idx] < SPIKE_LAST_LAY

    def bg_rate_forecast(self, d: dict, idx: np.ndarray) -> np.ndarray:
        return np.exp(bg_design(d, idx) @ self.beta + self.gamma[d["group"][idx], -1])


# ----------------------------------------------------------------------------- fitting
def _observation(d: dict, cutoff: int):
    """Per pipe: observed window [START, end) and event flag. Pipes that leaked before START are dropped.

    With d["is_victim"] (load_victims): a leak TRIGGERED by another pipe's leak ends the pipe's time at risk
    but is not a failure of its own (censored), so the hazard describes PRIMARY leaks only."""
    end_obs = float(cutoff + 1)
    keep = ~(d["leak_dec"] < START)
    left = keep & (d["leak_dec"] < end_obs)
    event = left & ~d.get("is_victim", np.zeros(d["n"], bool))
    end = np.where(left, d["leak_dec"], end_obs)
    return np.flatnonzero(keep), event, end


def load_victims(d: dict, root: Path) -> dict:
    """Mark leaks that scripts/leak_events.py attributes to another pipe's leak (role == victim)."""
    ev = pd.read_csv(root / "data/processed/leak_events.csv")
    vic = set(ev.loc[ev["role"] == "victim", "Pipe ID"])
    d = dict(d)
    d["is_victim"] = np.array([k in vic for k in d["pid"]])
    return d


def _segments(d: dict, idx: np.ndarray, end: np.ndarray, cutoff: int):
    """Pipe-year exposure segments for the background part."""
    seg_i, seg_y, seg_e = [], [], []
    for y in range(lc.FIRST_YEAR, cutoff + 1):
        lo, hi = max(float(y), START), float(y + 1)
        e = np.clip(np.minimum(end[idx], hi) - lo, 0.0, None)
        m = e > 0
        seg_i.append(idx[m]); seg_y.append(np.full(m.sum(), y - lc.FIRST_YEAR)); seg_e.append(e[m])
    return np.concatenate(seg_i), np.concatenate(seg_y), np.concatenate(seg_e)


def fit(d: dict, cutoff: int, extrapolate: bool = False, n_iter: int = 40, verbose: bool = False) -> Fit:
    idx, event, end = _observation(d, cutoff)
    lay = d["lay_dec"]
    global N_OFFSETS
    mode = "material" if MATERIAL_SPIKE else OFFSET_MODE
    if mode == "batch":
        mat = d["batch_code"]
        N_OFFSETS = len(d["batch_names"])
        ridge = BATCH_RIDGE
    else:
        mat = d["mat_code"] if mode == "material" else np.zeros(d["n"], dtype=np.int64)
        N_OFFSETS = N_MAT_OFFSETS
        ridge = SPIKE_RIDGE
    a0, a1 = START - lay, end - lay
    ev_idx = idx[event[idx]]
    ev_year = np.floor(d["leak_dec"][ev_idx]).astype(int) - lc.FIRST_YEAR
    spike_on = np.ones(d["n"], bool) if extrapolate else lay < SPIKE_LAST_LAY

    seg_i, seg_y, seg_e = _segments(d, idx, end, cutoff)
    Xs = bg_design(d, seg_i)
    gs = d["group"][seg_i]
    G, Y = len(GROUP_EDGES) + 1, cutoff - lc.FIRST_YEAR + 1
    Xe = bg_design(d, ev_idx)
    ge = d["group"][ev_idx]
    p = Xs.shape[1]

    sp = Spike()
    theta = np.zeros(p + G * Y)
    theta[p:] = np.log(0.01)
    sidx = idx[spike_on[idx]]

    def bg_log_rate_events(th):
        return Xe @ th[:p] + th[p:].reshape(G, Y)[ge, ev_year]

    prev = -np.inf
    for it in range(n_iter):
        # ---- E-step: share of each leak attributed to the spike
        log_hb = bg_log_rate_events(theta)
        log_hs = np.where(spike_on[ev_idx], spike_log_hazard(sp, lay[ev_idx], a1[ev_idx], mat[ev_idx]), -np.inf)
        r = np.exp(log_hs - np.logaddexp(log_hs, log_hb))

        # ---- M-step background: Poisson with events weighted 1 - r, random-walk penalty on gamma
        w_ev = 1.0 - r

        def nll_bg(th):
            beta, gam = th[:p], th[p:].reshape(G, Y)
            eta_s = Xs @ beta + gam[gs, seg_y]
            mu = seg_e * np.exp(eta_s)
            eta_e = Xe @ beta + gam[ge, ev_year]
            dg = np.diff(gam, axis=1)
            f = mu.sum() - (w_ev * eta_e).sum() + 0.5 * RW_PENALTY * (dg ** 2).sum() + 0.5 * RIDGE * (beta ** 2).sum()
            g_beta = Xs.T @ mu - Xe.T @ w_ev + RIDGE * beta
            g_gam = np.zeros((G, Y))
            np.add.at(g_gam, (gs, seg_y), mu)
            np.add.at(g_gam, (ge, ev_year), -w_ev)
            pen = np.zeros((G, Y))
            pen[:, 1:] += dg
            pen[:, :-1] -= dg
            g_gam += RW_PENALTY * pen
            return f, np.r_[g_beta, g_gam.ravel()]

        theta = minimize(nll_bg, theta, jac=True, method="L-BFGS-B").x

        # ---- M-step spike: survival terms for spike pipes + r-weighted log hazard at the leaks
        sp_ev = spike_on[ev_idx]
        L_ev, A_ev, M_ev, r_ev = lay[ev_idx][sp_ev], a1[ev_idx][sp_ev], mat[ev_idx][sp_ev], r[sp_ev]

        def nll_sp(v):
            s = Spike.of(v)
            ll = (spike_log_surv(s, lay[sidx], a1[sidx], mat[sidx])
                  - spike_log_surv(s, lay[sidx], a0[sidx], mat[sidx])).sum()
            ll += (r_ev * spike_log_hazard(s, L_ev, A_ev, M_ev)).sum()
            return -ll + 0.5 * ridge * float((s.dmu ** 2).sum() + (s.dpi ** 2).sum())

        on = mode != "none"
        off_bounds = [(-3.0, 3.0) if on else (0.0, 0.0)] * N_OFFSETS
        off_bounds += [(-4.0, 4.0) if on else (0.0, 0.0)] * N_OFFSETS

        sp = Spike.of(minimize(nll_sp, sp.vec(), method="L-BFGS-B",
                               bounds=[(50, 75), (-1.5, 0.5), (np.log(0.05), np.log(5)), (-5, 5), (-0.05, 0.05) if DRIFT_CURVATURE else (0.0, 0.0),
                                       (-5.0, 5.0) if PI_MODE == "step" else (-1.0, 1.0) if PI_MODE == "linear" else (0.0, 0.0)]
                               + off_bounds).x)

        # ---- observed-data log-likelihood (for convergence)
        log_hb = bg_log_rate_events(theta)
        log_hs = np.where(spike_on[ev_idx], spike_log_hazard(sp, lay[ev_idx], a1[ev_idx], mat[ev_idx]), -np.inf)
        beta, gam = theta[:p], theta[p:].reshape(G, Y)
        ll = (np.logaddexp(log_hs, log_hb).sum()
              - (seg_e * np.exp(Xs @ beta + gam[gs, seg_y])).sum()
              + (spike_log_surv(sp, lay[sidx], a1[sidx], mat[sidx])
                 - spike_log_surv(sp, lay[sidx], a0[sidx], mat[sidx])).sum())
        if verbose:
            print(f"  EM {it:2d}: loglik {ll:,.1f}  mu(1960)={sp.a:.2f} slope={sp.b:+.3f} "
                  f"sigma={sp.sigma:.2f} pi(1955)={sp.pi_at(1955):.3f} pi(1960)={sp.pi:.3f}")
        if abs(ll - prev) < 0.05:
            break
        prev = ll

    return Fit(sp, theta[:p], theta[p:].reshape(G, Y), np.arange(lc.FIRST_YEAR, cutoff + 1), cutoff, extrapolate, ll,
               off_code=mat.copy(), mode=mode)


def batch_offsets(f: "Fit", d: dict, top: int = 8) -> str:
    """The largest batch shifts (failure age in years, logit pi)."""
    s = f.spike
    order = np.argsort(-np.abs(s.dmu))[:top]
    return "  ".join(f"{d['batch_names'][k]}: age {s.dmu[k]:+.2f}y pi {s.dpi[k]:+.2f}" for k in order)


def material_offsets(s: Spike) -> str:
    return "  ".join(f"{m}: age {s.dmu[k]:+.2f}y logit(pi) {s.dpi[k]:+.2f}" for k, m in enumerate(lc.MATS[1:]))


# ----------------------------------------------------------------------------- forecasting
def yearly_p(f: Fit, d: dict, idx: np.ndarray, years) -> np.ndarray:
    """(len(idx), len(years)) matrix: P(leak during calendar year Y | not leaked by the start of Y)."""
    lay = d["lay_dec"][idx]
    mat = f.off_code[idx] if f.off_code is not None else d["mat_code"][idx]
    on = f.spike_mask(d, idx)
    hb = f.bg_rate_forecast(d, idx)
    out = []
    for Y in years:
        H = hb.copy()
        ds = spike_log_surv(f.spike, lay, Y - lay, mat) - spike_log_surv(f.spike, lay, Y + 1.0 - lay, mat)
        H += np.where(on, np.maximum(ds, 0.0), 0.0)
        out.append(-np.expm1(-H))
    return np.column_stack(out)


def write_risk(d: dict, out: Path, extrapolate: bool, verbose: bool = True) -> pd.DataFrame:
    c = lc.LAST_YEAR
    f = fit(d, c, extrapolate=extrapolate, verbose=verbose)
    at = np.flatnonzero(~(d["leak_year"] <= c))
    risk = lc.sub(d, ~(d["leak_year"] <= c))
    years = list(range(c + 1, c + 26))
    P = yearly_p(f, d, at, years)
    p10 = 1 - np.prod(1 - P[:, :10], axis=1)
    df = pd.DataFrame(P, columns=[f"p_{y}" for y in years])
    df.insert(0, "p_leak_annual", 1 - (1 - p10) ** 0.1)
    df.insert(0, "cost_if_leak", lc.cost_hat(lc.fit_cost(d, c), risk))
    df.insert(0, "Pipe ID", risk["pid"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    s = f.spike
    print(f"spike: failure age {s.a:.2f} at lay 1960, {s.b:+.3f}/lay-year, sd {s.sigma:.2f}; "
          f"defective share pi = {s.pi_at(1955):.3f} (1955), {s.pi:.3f} (1960), {s.pi_at(1965):.3f} (1965)")
    if f.mode == "material":
        print(f"material offsets (vs wrought iron): {material_offsets(s)}")
    elif f.mode == "batch":
        print(f"largest batch offsets: {batch_offsets(f, d)}")
    print(f"wrote {out} ({len(df)} pipes; expected leaks 2027={P[:, 0].sum():.0f}, 2028={P[:, 1].sum():.0f}, "
          f"2027-2036={p10.sum():.0f})")
    return df


# ----------------------------------------------------------------------------- holdout
def run_holdout(d: dict, cutoffs=(2021, 2022, 2023, 2024, 2025), extrapolate: bool = False) -> pd.DataFrame:
    rows = []
    ly = d["leak_year"]
    for c in cutoffs:
        f = fit(d, c, extrapolate=extrapolate)
        at = np.flatnonzero(~(ly <= c))
        risk = lc.sub(d, ~(ly <= c))
        H = lc.LAST_YEAR - c
        P = yearly_p(f, d, at, range(c + 1, c + 1 + H))
        cost = lc.cost_hat(lc.fit_cost(d, c), risk)
        act_cost = np.nan_to_num(risk["cost"])
        for k in range(H):
            Y = c + 1 + k
            alive = ~(risk["leak_year"] < Y)
            y = (risk["leak_year"][alive] == Y).astype(float)
            p = np.clip(P[alive, k], 1e-6, 1 - 1e-6)
            top = np.argsort(-p * cost[alive])[: len(p) // 10]
            rows.append(dict(cutoff=c, h=k + 1, pred=p.sum(), actual=y.sum(),
                             logloss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                             top10_leaks=float(y[top].sum() / max(y.sum(), 1)),
                             top10_usd=float((act_cost[alive] * y)[top].sum() / max((act_cost[alive] * y).sum(), 1))))
        s = f.spike
        print(f"cutoff {c}: mu(1960)={s.a:.2f} slope={s.b:+.3f} sd={s.sigma:.2f} pi(1955)={s.pi_at(1955):.3f} "
              f"pi(1960)={s.pi:.3f} pi(1965)={s.pi_at(1965):.3f}"
              + (f" | {material_offsets(s)}" if f.mode == "material" else
                 f" | {batch_offsets(f, d, 4)}" if f.mode == "batch" else ""), flush=True)
    out = pd.DataFrame(rows)
    out["ratio"] = out["pred"] / out["actual"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--write-risk", default=None)
    ap.add_argument("--extrapolate-spike", action="store_true",
                    help="also apply the drifting spike to the 1966-1985 batches (no spike observed there yet)")
    ap.add_argument("--material-spike", action="store_true",
                    help="per-material failure-age / share offsets (experimental, see MATERIAL_SPIKE)")
    ap.add_argument("--batch-spike", action="store_true",
                    help="per-batch (lay year x material) failure-age / share offsets, see OFFSET_MODE")
    ap.add_argument("--primary-only", action="store_true",
                    help="fit PRIMARY leaks only: leaks triggered by another pipe's leak (data/processed/leak_events.csv) "
                         "are censored; use with the trigger model (pipeplan.contagion)")
    args = ap.parse_args()
    global MATERIAL_SPIKE, OFFSET_MODE
    MATERIAL_SPIKE = args.material_spike
    OFFSET_MODE = "batch" if args.batch_spike else ("material" if args.material_spike else "none")
    root = Path(args.root)
    d = load(root)
    if args.primary_only:
        d = load_victims(d, root)
        print(f"primary-only fit: {int(d['is_victim'].sum())} triggered leaks censored")
    if args.write_risk:
        out = Path(args.write_risk)
        write_risk(d, out if out.is_absolute() else root / out, args.extrapolate_spike)
        return
    res = run_holdout(d, extrapolate=args.extrapolate_spike)
    pd.set_option("display.width", 200)
    print(res.groupby("h")[["ratio", "logloss", "top10_leaks", "top10_usd"]].mean().round(3).to_string())
    print(res[["logloss", "top10_leaks", "top10_usd"]].mean().round(4).to_string())
    print(res.pivot(index="cutoff", columns="h", values="ratio").round(2).to_string())


if __name__ == "__main__":
    main()
