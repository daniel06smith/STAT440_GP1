"""Split the 2019-2026 leaks into sporadic, cascading and clustered leaks (instructor's lecture: three leak types).

    python scripts/leak_events.py            # -> data/processed/leak_events.csv, data/processed/trigger_kernel.csv

Lecture slide:  Sporadic  -- located in a single pipe, depends on material and surface.
                Cascading -- propagates from a source pipe, depends on NETWORK distance from the source.
                Clustered -- starts from a source pipe, depends on EUCLIDEAN distance from the source.
Data: 1,118 timestamps (date + minute) are shared by 2-12 leaks (4,112 leaks). An event = one source leak plus the
leaks it triggers at the same instant.

Method (iterated 3x):
  1. kernel q(euclid bin, network bin) = P(a still non-poly pipe at that distance leaks at the same instant as a
     source). Estimated from (source, candidate) pairs only: every primary leak is a source (single leaks are sources
     that triggered nothing); candidates are non-poly pipes within 2 km that had not leaked yet.
  2. source of each event = argmax_s  log h_s(t) + sum_{j != s} log q(s, j), with h_s(t) the batch-wave hazard of
     s at that moment (spike + background). The others are its victims.
Network = pipes sharing an end point within 0.5 m; network distance in hops and in metres along the pipes.

Outputs
  leak_events.csv     Pipe ID, ts (YYYY-MM-DD HH:MM), event_id (-1 = single leak), role (sporadic | source | victim),
                      source_pipe (the event's source Pipe ID), d_euclid_m / d_net_m / d_hops (victim to source;
                      d_net_m/d_hops = inf when not connected)
  trigger_kernel.csv  euclid_lo, euclid_hi (m), net (bin label: "<=k hops" bins or "not connected"), candidates,
                      victims, q (= victims / candidates)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import batch_wave_hazard as bw  # noqa: E402
import leak_calibration as lc  # noqa: E402

TOL = 0.5  # m, end points closer than this are the same junction (coordinates are rounded ~0.1 m)
R_MAX = 2000.0
E_BINS = [0, 50, 100, 200, 300, 400, 600, 800, 1200, 2000]
H_BINS = [0, 2, 5, 10, 20, 40, 80, np.inf]  # hop bins; last column = not connected
H_LABELS = ["<=2 hops", "3-5", "6-10", "11-20", "21-40", "41-80", ">80", "not connected"]


def network(p: pd.DataFrame):
    n = len(p)
    L = np.hypot(p["GPS x2"] - p["GPS x1"], p["GPS y2"] - p["GPS y1"]).to_numpy()
    E = np.r_[p[["GPS x1", "GPS y1"]].to_numpy(), p[["GPS x2", "GPS y2"]].to_numpy()]
    pr = np.array(list(cKDTree(E).query_pairs(TOL)))
    a, b = pr[:, 0] % n, pr[:, 1] % n
    m = a != b
    a, b = a[m], b[m]
    H = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    H = H.maximum(H.T)
    Wm = sp.coo_matrix(((L[a] + L[b]) / 2, (a, b)), shape=(n, n)).tocsr()
    Wm = Wm.maximum(Wm.T)
    return H, Wm


def bins(de, dh):
    ei = np.clip(np.digitize(de, E_BINS) - 1, 0, len(E_BINS) - 2)
    hi = np.where(np.isfinite(dh), np.clip(np.digitize(dh, H_BINS) - 1, 0, len(H_BINS) - 2), len(H_BINS) - 1)
    return ei, hi


def main() -> None:
    p = pd.read_csv(ROOT / "data/raw/pipes.csv", parse_dates=["Lay date"])
    t = pd.read_csv(ROOT / "data/raw/train.csv")
    n = len(p)
    pid = {k: i for i, k in enumerate(p["Pipe ID"])}
    mx = ((p["GPS x1"] + p["GPS x2"]) / 2).to_numpy()
    my = ((p["GPS y1"] + p["GPS y2"]) / 2).to_numpy()
    H, Wm = network(p)
    t["ts"] = t["Date"] + " " + t["Time"]
    t["i"] = t["Pipe ID"].map(pid)
    tmin = pd.to_datetime(t["ts"]).astype("int64").to_numpy() / 60e9
    leak_min = np.full(n, np.inf)
    leak_min[t["i"].to_numpy()] = tmin
    nonpoly = (p["Material"] != "polyurethane").to_numpy()

    # batch-wave hazard of every leaked pipe at its leak moment (prior for being the source)
    d = bw.load(ROOT)
    f = bw.fit(d, 2026)
    pos = {k: i for i, k in enumerate(d["pid"])}
    li = np.array([pos[k] for k in t["Pipe ID"]])
    lay, age = d["lay_dec"][li], d["leak_dec"][li] - d["lay_dec"][li]
    on = f.spike_mask(d, li)
    log_hs = np.where(on, bw.spike_log_hazard(f.spike, lay, age), -np.inf)
    yr = np.clip(np.floor(d["leak_dec"][li]).astype(int) - lc.FIRST_YEAR, 0, f.gamma.shape[1] - 1)
    log_hb = bw.bg_design(d, li) @ f.beta + f.gamma[d["group"][li], yr]
    t["log_h"] = np.logaddexp(log_hs, log_hb)

    groups = t.groupby("ts")["i"].apply(np.array)
    events = groups[groups.map(len) > 1]
    # pairwise distances inside events (hops + euclid), computed once
    ev_D = {}
    for ts_, ix in events.items():
        dh = cg.dijkstra(H, directed=False, unweighted=True, indices=ix)[:, ix]
        de = np.hypot(mx[ix][:, None] - mx[ix][None], my[ix][:, None] - my[ix][None])
        ev_D[ts_] = (de, dh)
    log_h_of = dict(zip(t["i"], t["log_h"]))
    tree = cKDTree(np.c_[mx, my])

    # initial kernel: pooled pairs inside events (any member as source), flat floor elsewhere
    q = np.full((len(E_BINS) - 1, len(H_LABELS)), 1e-5)
    source_of = {}
    for it in range(3):
        # ---- 2. choose sources
        new_src = {}
        for ts_, ix in events.items():
            de, dh = ev_D[ts_]
            ei, hi = bins(de, dh)
            lq = np.log(np.maximum(q[ei, hi], 1e-7))
            np.fill_diagonal(lq, 0.0)
            score = np.array([log_h_of[i] for i in ix]) + lq.sum(1)
            new_src[ts_] = int(np.argmax(score))
        changed = sum(new_src[k] != source_of.get(k, -1) for k in new_src)
        source_of = new_src
        # ---- 1. kernel from (source, candidate) pairs
        victims_of = {ix[source_of[ts_]]: np.delete(ix, source_of[ts_]) for ts_, ix in events.items()}
        is_victim = np.zeros(n, bool)
        for v in victims_of.values():
            is_victim[v] = True
        sources = t["i"].to_numpy()[~is_victim[t["i"].to_numpy()]]
        num = np.zeros_like(q)
        den = np.zeros_like(q)
        for s in sources:
            cand = np.array(tree.query_ball_point((mx[s], my[s]), R_MAX))
            cand = cand[(cand != s) & nonpoly[cand] & (leak_min[cand] >= leak_min[s])]
            de = np.hypot(mx[cand] - mx[s], my[cand] - my[s])
            dh = cg.dijkstra(H, directed=False, unweighted=True, indices=s, limit=H_BINS[-2] + 1)[cand]
            ei, hi = bins(de, dh)
            np.add.at(den, (ei, hi), 1)
            vic = victims_of.get(s)
            if vic is not None:
                hit = np.isin(cand, vic)
                np.add.at(num, (ei[hit], hi[hit]), 1)
        q = (num + 0.5 * 1e-5 * 1000) / (den + 1000)  # tiny prior toward 1e-5 for empty cells
        print(f"iteration {it}: {changed} event sources changed; {int(num.sum())} victims attributed, "
              f"{len(sources)} sources", flush=True)

    # ---- outputs
    rows = []
    for eid, (ts_, ix) in enumerate(events.items()):
        de, dh = ev_D[ts_]
        s = source_of[ts_]
        src_pipe = p["Pipe ID"].iloc[ix[s]]
        dm = cg.dijkstra(Wm, directed=False, indices=ix[s])[ix]
        for k, i in enumerate(ix):
            rows.append(dict(**{"Pipe ID": p["Pipe ID"].iloc[i]}, ts=ts_, event_id=eid,
                             role="source" if k == s else "victim", source_pipe=src_pipe,
                             d_euclid_m=0.0 if k == s else de[s, k], d_net_m=0.0 if k == s else dm[k],
                             d_hops=0.0 if k == s else dh[s, k]))
    ev = pd.DataFrame(rows)
    single = t[~t["ts"].isin(events.index)]
    sing = pd.DataFrame({"Pipe ID": single["Pipe ID"], "ts": single["ts"], "event_id": -1, "role": "sporadic",
                         "source_pipe": single["Pipe ID"], "d_euclid_m": 0.0, "d_net_m": 0.0, "d_hops": 0.0})
    out = pd.concat([ev, sing], ignore_index=True)
    out.to_csv(ROOT / "data/processed/leak_events.csv", index=False)
    kt = [dict(euclid_lo=E_BINS[e], euclid_hi=E_BINS[e + 1], net=H_LABELS[h], candidates=int(den[e, h]),
               victims=int(num[e, h]), q=q[e, h]) for e in range(len(E_BINS) - 1) for h in range(len(H_LABELS))]
    pd.DataFrame(kt).to_csv(ROOT / "data/processed/trigger_kernel.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nroles:", out["role"].value_counts().to_dict())
    print("\nkernel q x 1000 (rows: euclidean m, cols: network hops from the source):")
    print(pd.DataFrame(q * 1000, index=[f"{a}-{b}" for a, b in zip(E_BINS[:-1], E_BINS[1:])], columns=H_LABELS)
          .round(3).to_string())
    print("\nvictims per cell:")
    print(pd.DataFrame(num, index=[f"{a}-{b}" for a, b in zip(E_BINS[:-1], E_BINS[1:])], columns=H_LABELS)
          .astype(int).to_string())
    v = out[out.role == "victim"]
    print("\nvictim distance to source: euclid median %.0f m, connected share %.2f, hops median (connected) %.0f"
          % (v.d_euclid_m.median(), np.isfinite(v.d_hops).mean(), v.d_hops[np.isfinite(v.d_hops)].median()))


if __name__ == "__main__":
    main()
