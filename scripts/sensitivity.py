"""How sensitive is the plan to the construction settings?

    python scripts/sensitivity.py            (takes ~5 minutes; uses all CPU cores)

Each row rebuilds circles with one setting changed, orders them with the same
rule, and reports the model-based expected score. Output:
    data/processed/sensitivity_construction.csv
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VARIANTS = {
    "default": {},
    "no time-aware penalty": {"time_aware": False},
    "16 merge partners": {"k_neigh": 16},
    "max radius 300 m": {"r_max": 300.0},
    "max radius 1000 m": {"r_max": 1000.0},
    "no optional copper/brass (h_opt~0)": {"h_opt": 0.001},
    "optional horizon 6 y": {"h_opt": 6.0},
    "optional horizon 18 y": {"h_opt": 18.0},
    "optional horizon 25 y": {"h_opt": 25.0},
    "safety margin 0.02 m": {"margin": 0.02},
}


def run(item):
    name, kw = item
    from pipeplan.construct import Builder, Params
    from pipeplan.ordering import smith_order
    from pipeplan.simulator import expected_score, prepare
    from pipeplan.world import load_world

    t = time.time()
    w = load_world()
    b = Builder(w, Params(verbose=False, **kw)).run()
    prep = prepare(w, b.circles())
    best, _ = smith_order(w, prep, iters=5, verbose=False)
    e = expected_score(w, best)
    return {
        "variant": name, "settings": str(kw), "circles": len(b.clusters),
        "optional_pipes_targeted": b.n_optional_taken,
        "fixed_cost": sum(c.F for c in b.clusters.values()),
        "spend": e["spend"], "expected_leak_cost": e["leak_cost"], "expected_score": e["score"],
        "last_year": 2027 + e["last_year"], "covers_all_iron": prep.covers_all_iron,
        "seconds": round(time.time() - t),
    }


if __name__ == "__main__":
    import pandas as pd

    with ProcessPoolExecutor(max_workers=5) as ex:
        rows = list(ex.map(run, VARIANTS.items()))
    df = pd.DataFrame(rows)
    out = ROOT / "data/processed/sensitivity_construction.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["settings"]).round(0).to_string(index=False))
