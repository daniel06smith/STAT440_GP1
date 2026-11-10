"""Triggered leaks (cascading / clustered, instructor's lecture) for the simulator and the planner.

A *primary* leak at a source pipe s makes every other still-non-polyurethane pipe j leak at the same instant with
probability  q(euclid distance s-j, whether s and j are connected through the network).  Victims do not trigger
further leaks (in 2019-2026 every multi-leak event has a single timestamp).

The kernel is fitted from data/processed/leak_events.csv (scripts/leak_events.py): every non-victim leak is a
source, candidates are pipes within the distance range that were still non-poly at that moment, victims are the
event's other leaks. Hop count matters little once straight-line distance and connectivity are known, so the
network enters only through connected-component labels (end points within 0.5 m = same junction).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
from scipy.spatial import cKDTree

E_BINS = np.array([0, 50, 100, 200, 300, 400, 600, 800, 1200, 2000, 3000, 4500, 7000], float)
TOL = 0.5


def components(p: pd.DataFrame) -> np.ndarray:
    """Connected-component label of every pipe (pipes sharing an end point within TOL metres)."""
    n = len(p)
    E = np.r_[p[["GPS x1", "GPS y1"]].to_numpy(), p[["GPS x2", "GPS y2"]].to_numpy()]
    pr = cKDTree(E).query_pairs(TOL, output_type="ndarray").reshape(-1, 2)  # (0, 2) when no junctions
    a, b = pr[:, 0] % n, pr[:, 1] % n
    m = a != b
    A = sp.coo_matrix((np.ones(m.sum()), (a[m], b[m])), shape=(n, n)).tocsr()
    return cg.connected_components(A, directed=False)[1]


@dataclass
class Kernel:
    q: np.ndarray  # (n_bins, 2): column 0 = not connected, 1 = connected
    candidates: np.ndarray
    victims: np.ndarray

    def __call__(self, de: np.ndarray, conn: np.ndarray) -> np.ndarray:
        ei = np.clip(np.searchsorted(E_BINS, de, side="right") - 1, 0, len(E_BINS) - 2)
        out = self.q[ei, conn.astype(np.int64)]
        return np.where(de < E_BINS[-1], out, 0.0)

    def table(self) -> pd.DataFrame:
        idx = [f"{a:.0f}-{b:.0f} m" for a, b in zip(E_BINS[:-1], E_BINS[1:])]
        return pd.DataFrame({"q_not_connected_x1000": self.q[:, 0] * 1000, "q_connected_x1000": self.q[:, 1] * 1000,
                             "cand_nc": self.candidates[:, 0], "cand_c": self.candidates[:, 1],
                             "victims_nc": self.victims[:, 0], "victims_c": self.victims[:, 1]}, index=idx)


def load_history(root: Path):
    p = pd.read_csv(root / "data/raw/pipes.csv", parse_dates=["Lay date"])
    ev = pd.read_csv(root / "data/processed/leak_events.csv")
    pid = {k: i for i, k in enumerate(p["Pipe ID"])}
    ev["i"] = ev["Pipe ID"].map(pid)
    ev["src"] = ev["source_pipe"].map(pid)
    ev["t"] = pd.to_datetime(ev["ts"]).astype("int64") / 60e9
    return p, ev


def fit_kernel(p: pd.DataFrame, ev: pd.DataFrame, comp: np.ndarray, before: float | None = None) -> Kernel:
    """q per (distance bin, connected) = victims / candidates. `before`: only leaks with t < before (minutes)."""
    if before is not None:
        ev = ev[ev["t"] < before]
    n = len(p)
    mx = ((p["GPS x1"] + p["GPS x2"]) / 2).to_numpy()
    my = ((p["GPS y1"] + p["GPS y2"]) / 2).to_numpy()
    nonpoly = (p["Material"] != "polyurethane").to_numpy()
    leak_t = np.full(n, np.inf)
    leak_t[ev["i"].to_numpy()] = ev["t"].to_numpy()
    victims = ev[ev["role"] == "victim"]
    vic_by_src = victims.groupby("src")["i"].apply(np.array).to_dict()
    sources = ev.loc[ev["role"] != "victim", ["i", "t"]].to_numpy()
    nb = len(E_BINS) - 1
    num = np.zeros((nb, 2))
    den = np.zeros((nb, 2))
    for s, ts in sources:
        s = int(s)
        alive = nonpoly & (leak_t >= ts)
        alive[s] = False
        j = np.flatnonzero(alive)
        de = np.hypot(mx[j] - mx[s], my[j] - my[s])
        ok = de < E_BINS[-1]
        j, de = j[ok], de[ok]
        ei = np.searchsorted(E_BINS, de, side="right") - 1
        c = (comp[j] == comp[s]).astype(np.int64)
        np.add.at(den, (ei, c), 1)
        v = vic_by_src.get(s)
        if v is not None:
            hit = np.isin(j, v)
            np.add.at(num, (ei[hit], c[hit]), 1)
    q = num / np.maximum(den, 1)
    return Kernel(q, den, num)


def expected_victims(kern: Kernel, p: pd.DataFrame, comp: np.ndarray, src: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Sum of q over the alive pipes, for each source pipe index in `src`."""
    mx = ((p["GPS x1"] + p["GPS x2"]) / 2).to_numpy()
    my = ((p["GPS y1"] + p["GPS y2"]) / 2).to_numpy()
    j = np.flatnonzero(alive)
    out = np.empty(len(src))
    for k, s in enumerate(src):
        de = np.hypot(mx[j] - mx[s], my[j] - my[s])
        qq = kern(de, comp[j] == comp[s])
        qq[j == s] = 0.0
        out[k] = qq.sum()
    return out


# ----------------------------------------------------------------------------- full trigger model
LAY_EDGES = [0, 1954, 1956, 1960, 1964, 1966, 3000]
LAY_GROUPS = ["<1954", "1954-55", "1956-59", "1960-63", "1964-65", "1966+"]
MATS = ["wrought iron", "cast iron", "gray iron", "copper", "brass"]
SURFS = ["grassland", "farmland", "swamp", "water", "road", "structure"]


def pipe_attributes(p: pd.DataFrame) -> dict:
    lay = p["Lay date"].dt.year.fillna(p.loc[p["Material"] == "wrought iron", "Lay date"].dt.year.median()).to_numpy()
    mat = p["Material"].fillna("wrought iron").to_numpy()
    return {
        "mx": ((p["GPS x1"] + p["GPS x2"]) / 2).to_numpy(), "my": ((p["GPS y1"] + p["GPS y2"]) / 2).to_numpy(),
        "lay_year": np.floor(lay).astype(int),
        "lay_group": np.searchsorted(LAY_EDGES, lay, side="right") - 1,
        "mat": np.array([MATS.index(m) if m in MATS else -1 for m in mat]),  # -1 = polyurethane
        "surf": np.array([SURFS.index(s) for s in p["Surface"]]),
    }


@dataclass
class TriggerModel:
    """P(j is a victim | primary leak at s) = kernel(d, connected) * exp(eta_s + eta_j + eta_sj)  (capped at 1)."""
    kernel: Kernel
    b_const: float
    b_vmat: np.ndarray    # per MATS
    b_vsurf: np.ndarray   # per SURFS
    b_vlay: np.ndarray    # per LAY_GROUPS
    b_smat: np.ndarray
    b_slay: np.ndarray
    b_same_mat: float
    b_same_year: float
    b_same_group: float

    def source_factor(self, A: dict, s: int) -> float:
        return float(np.exp(self.b_const + self.b_smat[A["mat"][s]] + self.b_slay[A["lay_group"][s]]))

    def victim_factor(self, A: dict, j: np.ndarray) -> np.ndarray:
        return np.exp(self.b_vmat[A["mat"][j]] + self.b_vsurf[A["surf"][j]] + self.b_vlay[A["lay_group"][j]])

    def probs(self, A: dict, comp: np.ndarray, s: int, j: np.ndarray, vf: np.ndarray | None = None) -> np.ndarray:
        """Victim probabilities of candidate pipes j (non-poly, alive) for a primary leak at s."""
        de = np.hypot(A["mx"][j] - A["mx"][s], A["my"][j] - A["my"][s])
        k = self.kernel(de, comp[j] == comp[s])
        pair = (self.b_same_mat * (A["mat"][j] == A["mat"][s]) + self.b_same_year * (A["lay_year"][j] == A["lay_year"][s])
                + self.b_same_group * (A["lay_group"][j] == A["lay_group"][s]))
        vf = self.victim_factor(A, j) if vf is None else vf
        q = k * self.source_factor(A, s) * vf * np.exp(pair)
        q[j == s] = 0.0
        return np.minimum(q, 1.0)

    def to_json(self) -> dict:
        return {"kernel_q": self.kernel.q.tolist(), "kernel_bins_m": E_BINS.tolist(),
                **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if k != "kernel"}}


def fit_trigger_model(p: pd.DataFrame, ev: pd.DataFrame, comp: np.ndarray, n_sources: int = 6000,
                      control_rate: float = 0.02, seed: int = 0, before: float | None = None) -> TriggerModel:
    """Weighted Poisson (rare-event Bernoulli) on (source, candidate) pairs: offset log kernel, case-control sample."""
    from scipy.optimize import minimize
    if before is not None:
        ev = ev[ev["t"] < before]
    kern = fit_kernel(p, ev, comp)
    A = pipe_attributes(p)
    n = len(p)
    nonpoly = A["mat"] >= 0
    leak_t = np.full(n, np.inf)
    leak_t[ev["i"].to_numpy()] = ev["t"].to_numpy()
    vic_by_src = ev[ev["role"] == "victim"].groupby("src")["i"].apply(np.array).to_dict()
    srcs = ev.loc[ev["role"] != "victim", ["i", "t"]].to_numpy()
    rng = np.random.default_rng(seed)
    if len(srcs) > n_sources:
        srcs = srcs[rng.choice(len(srcs), n_sources, replace=False)]
    cols = []
    for s, tm in srcs:
        s = int(s)
        alive = nonpoly & (leak_t >= tm)
        alive[s] = False
        j = np.flatnonzero(alive)
        de = np.hypot(A["mx"][j] - A["mx"][s], A["my"][j] - A["my"][s])
        k = kern(de, comp[j] == comp[s])
        hit = np.isin(j, vic_by_src.get(s, np.array([], int)))
        keep = (k > 0) & (hit | (rng.random(len(j)) < control_rate))
        j, k, hit = j[keep], k[keep], hit[keep]
        cols.append((np.full(len(j), s), j, k, hit))
    S = np.concatenate([c[0] for c in cols]); J = np.concatenate([c[1] for c in cols])
    K = np.concatenate([c[2] for c in cols]); Y = np.concatenate([c[3] for c in cols]).astype(float)
    W = np.where(Y > 0, 1.0, 1.0 / control_rate)
    nm, ns, nl = len(MATS), len(SURFS), len(LAY_GROUPS)
    blocks = [np.ones((len(S), 1))]
    for arr, size in ((A["mat"][J], nm), (A["surf"][J], ns), (A["lay_group"][J], nl), (A["mat"][S], nm), (A["lay_group"][S], nl)):
        blocks.append(np.eye(size)[arr])
    blocks += [(A["mat"][J] == A["mat"][S])[:, None], (A["lay_year"][J] == A["lay_year"][S])[:, None],
               (A["lay_group"][J] == A["lay_group"][S])[:, None]]
    X = np.hstack(blocks).astype(float)
    off = np.log(np.maximum(K, 1e-12))
    pen = np.r_[0.0, np.full(X.shape[1] - 1, 1.0)]  # ridge on everything but the intercept (dummy sets are full)

    def f(b):
        eta = off + X @ b
        mu = np.exp(eta)
        return float((W * (mu - Y * eta)).sum() + 0.5 * (pen * b * b).sum()), X.T @ (W * (mu - Y)) + pen * b

    b = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
    o = 1
    def take(k):
        nonlocal o
        v = b[o:o + k]; o += k; return v
    vm, vs, vl, sm, sl = take(nm), take(ns), take(nl), take(nm), take(nl)
    same = take(3)
    return TriggerModel(kern, float(b[0]), vm, vs, vl, sm, sl, float(same[0]), float(same[1]), float(same[2]))


def prob_matrix(tm: TriggerModel, A: dict, comp: np.ndarray, S: np.ndarray, J: np.ndarray,
                vf: np.ndarray | None = None) -> np.ndarray:
    """(len S, len J) victim probabilities for primary leaks at each S (vectorised TriggerModel.probs), float32."""
    de = np.hypot(A["mx"][S][:, None] - A["mx"][J][None, :], A["my"][S][:, None] - A["my"][J][None, :]).astype(np.float32)
    ei = np.clip(np.searchsorted(E_BINS, de, side="right") - 1, 0, len(E_BINS) - 2)
    conn = comp[S][:, None] == comp[J][None, :]
    k = tm.kernel.q[ei, conn.astype(np.int64)].astype(np.float32)
    k[de >= E_BINS[-1]] = 0.0
    pair = (tm.b_same_mat * (A["mat"][S][:, None] == A["mat"][J][None, :])
            + tm.b_same_year * (A["lay_year"][S][:, None] == A["lay_year"][J][None, :])
            + tm.b_same_group * (A["lay_group"][S][:, None] == A["lay_group"][J][None, :])).astype(np.float32)
    sf = np.exp(tm.b_const + tm.b_smat[A["mat"][S]] + tm.b_slay[A["lay_group"][S]]).astype(np.float32)
    vf = tm.victim_factor(A, J) if vf is None else vf
    q = k * np.exp(pair) * sf[:, None] * vf[None, :].astype(np.float32)
    q[S[:, None] == J[None, :]] = 0.0
    return np.minimum(q, 1.0)
