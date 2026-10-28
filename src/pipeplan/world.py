"""The Star Kingdom "world": every pipe, its geometry, and its leak risk.

Everything the simulator and the circle builder need to know about the
network lives in the `World` object, stored as flat numpy arrays (one entry
per pipe, in the row order of ``pipes.csv``).

Rules taken from the professor's project description (see Project_Description.md)
------------------------------------------------------------------------------
* work-unit cost = 100 + (surface_rate * area) ** 0.85 + 200 * metres_of_non_poly_pipe
* surface_rate  = sum of the rates of the *distinct* surface types among ALL pipes
  wholly inside the circle (polyurethane pipes and already-leaked pipes included)
* a pipe is "wholly inside" a closed circle iff BOTH of its end points are inside
* pipes that leaked in 2019-2026 are already polyurethane when the project starts
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Constants from the project description
# --------------------------------------------------------------------------
SURFACE_RATE = {  # $ per m^2
    "grassland": 3.0,
    "farmland": 7.0,
    "swamp": 12.0,
    "road": 17.0,
    "water": 32.0,
    "structure": 50.0,
}
SURFACES = list(SURFACE_RATE)  # position in this list = integer surface code
BASE_FEE = 100.0  # $ per work unit
AREA_EXPONENT = 0.85
LINEAR_RATE = 200.0  # $ per metre of non-polyurethane pipe replaced
ANNUAL_BUDGET = 10_000_000.0
N_YEARS = 25

# surface_rate for every possible *set* of surfaces, indexed by a 6-bit mask
RATE_BY_MASK = np.array(
    [
        sum(SURFACE_RATE[s] for i, s in enumerate(SURFACES) if mask >> i & 1)
        for mask in range(1 << len(SURFACES))
    ]
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class World:
    """Flat, vectorised description of the pipe network (one entry per pipe)."""

    pipe_id: np.ndarray  # object, "P01001"
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    mx: np.ndarray  # mid point
    my: np.ndarray
    length: np.ndarray  # metres
    surf_bit: np.ndarray  # int64, 1 << surface code (for fast set-union via bitwise OR)
    material: np.ndarray  # object (missing material filled with "wrought iron")
    material_missing: np.ndarray  # bool: the 8 pipes whose material is blank in pipes.csv
    nonpoly0: np.ndarray  # bool: still non-polyurethane on 1 Jan 2027
    required: np.ndarray  # bool: iron pipe that MUST be inside some circle
    p_annual: np.ndarray  # P(leak in a given year) for pipes with nonpoly0 (else 0)
    lam: np.ndarray  # continuous hazard = -ln(1 - p_annual)
    leak_cost: np.ndarray  # E[cost | leak] in $
    loss_rate: np.ndarray  # lam * leak_cost: expected $ lost per year while unreplaced
    tree: cKDTree  # KD-tree over pipe MID points

    @property
    def n(self) -> int:
        return len(self.pipe_id)

    # ------------------------------------------------------------------
    def members(self, cx: float, cy: float, r: float) -> np.ndarray:
        """Indices of ALL pipes (any material) wholly inside the closed circle.

        A pipe is inside iff both end points are within distance r of (cx, cy).
        The mid point of an inside pipe is also inside (a disc is convex), so a
        ball query on mid points gives a complete candidate list.
        """
        cand = self.tree.query_ball_point((cx, cy), r)
        if not cand:
            return np.empty(0, dtype=np.int64)
        cand = np.asarray(cand, dtype=np.int64)
        r2 = r * r + 1e-9
        ok = ((self.x1[cand] - cx) ** 2 + (self.y1[cand] - cy) ** 2 <= r2) & (
            (self.x2[cand] - cx) ** 2 + (self.y2[cand] - cy) ** 2 <= r2
        )
        return cand[ok]

    def surface_rate(self, members: np.ndarray) -> float:
        """Sum of the rates of the distinct surface types among `members`."""
        if len(members) == 0:
            return 0.0
        return float(RATE_BY_MASK[np.bitwise_or.reduce(self.surf_bit[members])])


def build_world(
    pipes: pd.DataFrame, leaked_ids: set, risk: pd.DataFrame | None
) -> World:
    """Build a World from the raw tables.

    pipes      : pipes.csv as a DataFrame
    leaked_ids : set of Pipe IDs that appear in train.csv (leaked 2019-2026)
    risk       : table with columns ``Pipe ID, p_annual, leak_cost`` for every pipe
                 that is still non-polyurethane in 2027 (None -> zero risk)
    """
    pipe_id = pipes["Pipe ID"].to_numpy(dtype=object)
    x1 = pipes["GPS x1"].to_numpy(float)
    y1 = pipes["GPS y1"].to_numpy(float)
    x2 = pipes["GPS x2"].to_numpy(float)
    y2 = pipes["GPS y2"].to_numpy(float)

    material_missing = pipes["Material"].isna().to_numpy()
    # 8 pipes have a blank material. We cannot rule out that they are iron and a
    # single uncovered iron pipe gets the whole submission rejected, so they are
    # treated as (wrought) iron -- the same choice Cheryl, Kenson and Daniel made.
    material = pipes["Material"].fillna("wrought iron").to_numpy(dtype=object)

    leaked = np.fromiter((p in leaked_ids for p in pipe_id), dtype=bool, count=len(pipe_id))
    nonpoly0 = (material != "polyurethane") & ~leaked
    is_iron = np.array(["iron" in m for m in material])
    required = nonpoly0 & is_iron

    surf_code = pipes["Surface"].map({s: i for i, s in enumerate(SURFACES)})
    if surf_code.isna().any():
        raise ValueError("unknown surface type in pipes.csv")
    surf_bit = (1 << surf_code.to_numpy(np.int64)).astype(np.int64)

    p_annual = np.zeros(len(pipe_id))
    leak_cost = np.zeros(len(pipe_id))
    if risk is not None:
        r = risk.set_index("Pipe ID")
        idx = pd.Series(np.arange(len(pipe_id)), index=pipe_id)
        have = idx.reindex(r.index)
        if have.isna().any():
            raise ValueError("risk table contains unknown Pipe IDs")
        p_annual[have.to_numpy(np.int64)] = r["p_annual"].to_numpy(float)
        leak_cost[have.to_numpy(np.int64)] = r["leak_cost"].to_numpy(float)
        missing = nonpoly0 & (p_annual == 0)
        if missing.any():
            raise ValueError(f"{missing.sum()} non-poly pipes have no leak-risk estimate")
    # pipes that are already polyurethane never leak again (project notes)
    p_annual[~nonpoly0] = 0.0
    leak_cost[~nonpoly0] = 0.0
    lam = -np.log1p(-p_annual)

    return World(
        pipe_id=pipe_id,
        x1=x1, y1=y1, x2=x2, y2=y2,
        mx=(x1 + x2) / 2, my=(y1 + y2) / 2,
        length=np.hypot(x2 - x1, y2 - y1),
        surf_bit=surf_bit,
        material=material,
        material_missing=material_missing,
        nonpoly0=nonpoly0,
        required=required,
        p_annual=p_annual,
        lam=lam,
        leak_cost=leak_cost,
        loss_rate=lam * leak_cost,
        tree=cKDTree(np.column_stack([(x1 + x2) / 2, (y1 + y2) / 2])),
    )


def load_world(root: Path | None = None) -> World:
    """Load the real data: data/raw/{pipes,train}.csv + data/processed/leak_risk_cheryl.csv."""
    root = Path(root) if root else REPO_ROOT
    pipes = pd.read_csv(root / "data/raw/pipes.csv")
    leaks = pd.read_csv(root / "data/raw/train.csv")
    risk = pd.read_csv(root / "data/processed/leak_risk_cheryl.csv").rename(
        columns={"p_leak_annual": "p_annual", "cost_if_leak": "leak_cost"}
    )
    return build_world(pipes, set(leaks["Pipe ID"]), risk[["Pipe ID", "p_annual", "leak_cost"]])
