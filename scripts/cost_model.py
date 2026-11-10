"""Cost-given-leak model with engineered features + rolling-origin holdout.

    python scripts/cost_model.py                                      # holdout report only
    python scripts/cost_model.py --write data/processed/cost_model_v2.csv --model glm_full
    python scripts/cost_model.py --write ... --attach data/processed/leak_risk_ensemble_w050.csv

Why: road + structure leaks are ~81% of 2019-2026 leak dollars, and the old cost model
(leak_calibration.fit_cost: Gamma GLM on material + surface + log length, main effects only) gives
almost one number per surface. Within road, the median cost ranges $14k (copper) to $93k
(wrought iron), and log cost also moves with age at leak and location.

Models compared (all predict E[cost | leak], the MEAN, because the planner multiplies it by P(leak)):
  glm_base   the existing model (material + surface + log length)
  glm_inter  + material x surface interactions + age at leak (+ age^2)
  glm_full   glm_inter + spatial kNN feature (shrunken mean log-cost residual of nearby past leaks)
  lgbm       LightGBM, gamma objective, on the same raw features + x/y
  lognorm    log cost ~ material + surface + log length (OLS), surface-specific sigma;
             E[cost] = exp(eta + sigma_s^2 / 2). Same mean target, but estimated from the whole shape of the
             distribution, so one $7M leak moves it far less than it moves a sample mean.
  lognorm_ms lognorm + material x surface interactions (ridge)

Level check (--loyo): leave-one-YEAR-out, 8 non-overlapping test folds, so the pred/actual $ ratio is not
dominated by the same 2026 leaks in every fold (as in the rolling holdout). With heavy tails even a correct
mean over-predicts most folds; what matters is the pooled ratio over all folds.

Rolling holdout: for each cutoff c, fit on leaks dated <= c and score leaks dated > c. Ages for the test
leaks are taken at c + 1.5 (mid of the next year), NOT at the true leak date -- that is what the model
will see when it is used for planning, so the check is honest about that.

Output schema (--write): Pipe ID (str), cost_if_leak ($, E[cost | leak] at age mid-2027), one row per
pipe that is still non-polyurethane in 2027 (27,248). --attach copies an existing leak_risk_*.csv and
replaces its cost_if_leak column (writes <name>_costv2.csv next to it), so build_plan.py can use it as is.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MATS = ["wrought iron", "cast iron", "gray iron", "copper", "brass"]
SURFS = ["grassland", "farmland", "swamp", "water", "road", "structure"]
DEPLOY_T = 2027.5
K_NN = 15        # neighbours used by the spatial feature
NN_SHRINK = 10.0  # pseudo-count shrinking the neighbour mean toward 0
RIDGE = 1.0


# ----------------------------------------------------------------------------- data
def dec(ts: pd.Series) -> pd.Series:
    return ts.dt.year + (ts.dt.dayofyear - 0.5) / 365.25


def load(root: Path) -> pd.DataFrame:
    """One row per non-polyurethane pipe (original material), with leak date/cost if it leaked."""
    p = pd.read_csv(root / "data/raw/pipes.csv", parse_dates=["Lay date"])
    t = pd.read_csv(root / "data/raw/train.csv", parse_dates=["Date"])
    p["Material"] = p["Material"].fillna("wrought iron")  # the 8 blank-material pipes
    p = p[p["Material"] != "polyurethane"].reset_index(drop=True)
    lay = dec(p["Lay date"])
    p["lay"] = lay.fillna(lay[p["Material"] == "wrought iron"].median())  # 15 blank lay dates
    p["length"] = np.hypot(p["GPS x2"] - p["GPS x1"], p["GPS y2"] - p["GPS y1"])
    p["x"] = (p["GPS x1"] + p["GPS x2"]) / 2
    p["y"] = (p["GPS y1"] + p["GPS y2"]) / 2
    t = t.set_index("Pipe ID")
    p["leak_t"] = dec(p["Pipe ID"].map(t["Date"]))
    p["cost"] = p["Pipe ID"].map(t["Cost"])
    return p


# ----------------------------------------------------------------------------- spatial feature
def knn_feature(ref: pd.DataFrame, ref_resid: np.ndarray, qry: pd.DataFrame, loo: bool) -> np.ndarray:
    """Shrunken mean of the log-cost residuals of the K nearest reference leaks ON THE SAME SURFACE.

    Residuals are relative to the reference set's material x surface mean log cost, so the feature
    carries location only, not the material/surface mix around a pipe. loo=True when qry is ref itself
    (training rows): each row's own leak is excluded.
    """
    out = np.zeros(len(qry))
    for s in SURFS:
        rm = (ref["Surface"] == s).to_numpy()
        qm = (qry["Surface"] == s).to_numpy()
        if not qm.any() or rm.sum() < 2:
            continue
        tree = cKDTree(ref.loc[rm, ["x", "y"]].to_numpy())
        k = min(K_NN + int(loo), int(rm.sum()))
        _, idx = tree.query(qry.loc[qm, ["x", "y"]].to_numpy(), k=k)
        idx = np.atleast_2d(idx)
        if loo:
            idx = idx[:, 1:]  # nearest neighbour of a reference point is itself
        r = ref_resid[rm][idx]
        out[qm] = r.sum(1) / (r.shape[1] + NN_SHRINK)
    return out


def cell_mean_resid(train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """log cost minus the (shrunken) material x surface mean log cost."""
    lc = np.log(train["cost"].to_numpy())
    g = pd.DataFrame({"m": train["Material"].to_numpy(), "s": train["Surface"].to_numpy(), "lc": lc})
    surf_mean = g.groupby("s")["lc"].mean()
    cell = g.groupby(["m", "s"])["lc"].agg(["sum", "count"])
    cell_mean = {(m, s): (row["sum"] + 5 * surf_mean[s]) / (row["count"] + 5) for (m, s), row in cell.iterrows()}
    base = np.array([cell_mean[(m, s)] for m, s in zip(g["m"], g["s"])])
    return lc - base, cell_mean


# ----------------------------------------------------------------------------- GLM
def design(df: pd.DataFrame, age: np.ndarray, nn: np.ndarray | None, kind: str) -> np.ndarray:
    mat, surf = df["Material"].to_numpy(), df["Surface"].to_numpy()
    cols = [np.ones(len(df))]
    cols += [(mat == m).astype(float) for m in MATS[1:]]
    cols += [(surf == s).astype(float) for s in SURFS[1:]]
    cols.append(np.log(np.maximum(df["length"].to_numpy(), 0.5)) - 2.5)
    if kind in ("glm_inter", "glm_full"):
        for s in SURFS[1:]:
            for m in MATS[1:]:
                cols.append(((mat == m) & (surf == s)).astype(float))
        a = (np.clip(age, 50, 75) - 63.0) / 5.0  # clipped: the planner extrapolates ages past the data
        cols += [a, a * a]
        for s in ("road", "structure"):
            cols.append(a * (surf == s))
    if kind in ("glm_lay", "glm_lay_age"):
        # installation-batch group of the leaking pipe: the future is dominated by the 1964-65 spike and the
        # 1966+ batches, whose 2019-2026 leaks cost 0.83x / 0.67x what the batch-blind model predicts
        lay = df["lay"].to_numpy()
        for lo, hi in ((0, 1954), (1954, 1956), (1956, 1960), (1964, 1966), (1966, 3000)):  # base: 1960-63
            cols.append(((lay >= lo) & (lay < hi)).astype(float))
    if kind == "glm_lay_age":
        a = (np.clip(age, 50, 75) - 63.0) / 5.0
        cols += [a]
    if kind == "lognorm_ms":
        for s in SURFS[1:]:
            for m in MATS[1:]:
                cols.append(((mat == m) & (surf == s)).astype(float))
    if kind == "glm_full":
        cols.append(nn)
        for s in ("road", "structure"):
            cols.append(nn * (surf == s))
    return np.column_stack(cols)


def fit_gamma(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Gamma GLM, log link, IRLS with a ridge penalty (intercept unpenalised)."""
    beta = np.zeros(X.shape[1]); beta[0] = np.log(y.mean())
    P = ridge * np.eye(X.shape[1]); P[0, 0] = 0.0
    for _ in range(60):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        new = np.linalg.solve(X.T @ X + P, X.T @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


# ----------------------------------------------------------------------------- model wrapper
class CostModel:
    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, train: pd.DataFrame):
        self.train = train
        self.resid, self.cell_mean = cell_mean_resid(train)
        age = (train["leak_t"] - train["lay"]).to_numpy()
        nn = knn_feature(train, self.resid, train, loo=True)
        y = train["cost"].to_numpy()
        if self.kind.startswith("glm"):
            self.beta = fit_gamma(design(train, age, nn, self.kind), y)
        elif self.kind.startswith("lognorm"):
            X = design(train, age, nn, self.kind)
            P = RIDGE * np.eye(X.shape[1]); P[0, 0] = 0.0
            self.beta = np.linalg.solve(X.T @ X + P, X.T @ np.log(y))
            res = np.log(y) - X @ self.beta
            surf = train["Surface"].to_numpy()
            self.sigma2 = {s: float(res[surf == s].var()) for s in SURFS}
        else:
            import lightgbm as lgb
            params = dict(objective="gamma", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                          bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                          verbose=-1, seed=0)
            self.lgb = lgb.train(params, lgb.Dataset(self._lgb_X(train, age, nn), y), num_boost_round=400)
        return self

    def _lgb_X(self, df, age, nn):
        return pd.DataFrame({
            "mat": pd.Categorical(df["Material"], categories=MATS),
            "surf": pd.Categorical(df["Surface"], categories=SURFS),
            "age": age, "lay": df["lay"].to_numpy(), "loglen": np.log(np.maximum(df["length"].to_numpy(), 0.5)),
            "x": df["x"].to_numpy(), "y": df["y"].to_numpy(), "nn": nn,
        })

    def predict(self, df: pd.DataFrame, at_t: float) -> np.ndarray:
        age = at_t - df["lay"].to_numpy()
        nn = knn_feature(self.train, self.resid, df, loo=False)
        if self.kind.startswith("glm"):
            return np.exp(np.clip(design(df, age, nn, self.kind) @ self.beta, -30, 30))
        if self.kind.startswith("lognorm"):
            s2 = df["Surface"].map(self.sigma2).to_numpy()
            return np.exp(np.clip(design(df, age, nn, self.kind) @ self.beta + s2 / 2, -30, 30))
        return self.lgb.predict(self._lgb_X(df, age, nn))


# ----------------------------------------------------------------------------- holdout
def gamma_dev(y, mu):
    return float(np.mean(2 * (-np.log(y / mu) + (y - mu) / mu)))


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def run_holdout(p: pd.DataFrame, kinds, cutoffs=(2021, 2022, 2023, 2024, 2025)) -> pd.DataFrame:
    leaked = p[p["cost"].notna()].reset_index(drop=True)
    rows = []
    for c in cutoffs:
        tr = leaked[leaked["leak_t"] < c + 1].reset_index(drop=True)
        te = leaked[leaked["leak_t"] >= c + 1].reset_index(drop=True)
        y = te["cost"].to_numpy()
        for k in kinds:
            mu = CostModel(k).fit(tr).predict(te, at_t=c + 1.5)
            row = {"cutoff": c, "model": k, "n_test": len(te),
                   "gamma_dev": gamma_dev(y, mu), "spearman": spearman(mu, y),
                   "pred_over_actual_usd": mu.sum() / y.sum()}
            for s in ("road", "structure"):
                m = (te["Surface"] == s).to_numpy()
                row[f"spearman_{s}"] = spearman(mu[m], y[m])
                row[f"pred_over_actual_{s}"] = mu[m].sum() / y[m].sum()
            # planning-relevant: of the dollars in the top-10% predicted-cost leaks, how much is real?
            top = mu >= np.quantile(mu, 0.9)
            row["top10_share_actual_usd"] = y[top].sum() / y.sum()
            rows.append(row)
    return pd.DataFrame(rows)


def run_loyo(p: pd.DataFrame, kinds) -> pd.DataFrame:
    """Leave-one-year-out: fit on the other 7 years, predict that year's leaks at their true age."""
    leaked = p[p["cost"].notna()].reset_index(drop=True)
    yr = np.floor(leaked["leak_t"]).astype(int)
    rows = []
    for Y in sorted(yr.unique()):
        tr, te = leaked[yr != Y].reset_index(drop=True), leaked[yr == Y].reset_index(drop=True)
        y = te["cost"].to_numpy()
        for k in kinds:
            mu = _predict_true_age(CostModel(k).fit(tr), te)
            row = {"test_year": Y, "model": k, "n_test": len(te), "gamma_dev": gamma_dev(y, mu),
                   "pred_usd": mu.sum(), "actual_usd": y.sum()}
            for s in SURFS:
                m = (te["Surface"] == s).to_numpy()
                row[f"pred_{s}"], row[f"actual_{s}"] = mu[m].sum(), y[m].sum()
            rows.append(row)
    return pd.DataFrame(rows)


def _predict_true_age(model: "CostModel", df: pd.DataFrame) -> np.ndarray:
    """predict() evaluates age at one date; here each test leak uses its own leak date."""
    out = np.empty(len(df))
    t = df["leak_t"].to_numpy()
    for tt in np.unique(np.round(t, 2)):
        m = np.round(t, 2) == tt
        out[m] = model.predict(df[m].reset_index(drop=True), at_t=tt)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="glm_base,glm_inter,glm_full,lgbm")
    ap.add_argument("--write", type=Path, help="write Pipe ID, cost_if_leak for the 2027 non-poly pipes")
    ap.add_argument("--model", default="glm_full", help="model used by --write")
    ap.add_argument("--attach", type=Path, nargs="*", default=[],
                    help="leak_risk_*.csv files whose cost_if_leak gets replaced (-> *_costv2.csv)")
    ap.add_argument("--report", type=Path, default=Path("data/processed/cost_model_holdout.csv"))
    ap.add_argument("--skip-holdout", action="store_true")
    ap.add_argument("--loyo", action="store_true", help="leave-one-year-out level check")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    p = load(root)

    if not args.skip_holdout:
        res = run_holdout(p, args.models.split(","))
        out = args.report if args.report.is_absolute() else root / args.report
        res.to_csv(out, index=False)
        pd.set_option("display.width", 200)
        print(res.round(3).to_string(index=False))
        print("\nmean over cutoffs:")
        print(res.drop(columns=["cutoff", "n_test"]).groupby("model", sort=False).mean().round(3).to_string())

    if args.loyo:
        kinds = args.models.split(",")
        lo = run_loyo(p, kinds)
        lo.to_csv(root / "data/processed/cost_model_loyo.csv", index=False)
        lo["ratio"] = lo.pred_usd / lo.actual_usd
        print("\n=== leave-one-year-out: pred/actual $ per test year ===")
        print(lo.pivot(index="test_year", columns="model", values="ratio")[kinds].round(2).to_string())
        g = lo.groupby("model", sort=False)
        pooled = pd.DataFrame({"pooled_ratio": g.pred_usd.sum() / g.actual_usd.sum(),
                               "median_fold_ratio": g.apply(lambda x: (x.pred_usd / x.actual_usd).median()),
                               "gamma_dev": g.gamma_dev.mean()})
        for s in ("road", "structure", "water", "grassland"):
            pooled[f"pooled_{s}"] = g[f"pred_{s}"].sum() / g[f"actual_{s}"].sum()
        print("\n=== pooled over the 8 folds (1.00 = right level) ===")
        print(pooled.loc[kinds].round(3).to_string())

    if args.write:
        leaked = p[p["cost"].notna()].reset_index(drop=True)
        todo = p[p["cost"].isna()].reset_index(drop=True)  # still non-poly in 2027
        mu = CostModel(args.model).fit(leaked).predict(todo, at_t=DEPLOY_T)
        df = pd.DataFrame({"Pipe ID": todo["Pipe ID"], "cost_if_leak": mu})
        w = args.write if args.write.is_absolute() else root / args.write
        df.to_csv(w, index=False)
        print(f"\nwrote {w} ({len(df)} pipes); mean cost_if_leak by surface:")
        print(df.assign(s=todo["Surface"]).groupby("s")["cost_if_leak"].describe()[["mean", "50%", "min", "max"]].round(0))
        for a in args.attach:
            a = a if a.is_absolute() else root / a
            r = pd.read_csv(a)
            r["cost_if_leak"] = r["Pipe ID"].map(df.set_index("Pipe ID")["cost_if_leak"])
            assert r["cost_if_leak"].notna().all(), f"{a}: pipes without a new cost"
            dst = a.with_name(a.stem + "_costv2.csv")
            r.to_csv(dst, index=False)
            print(f"wrote {dst}")


if __name__ == "__main__":
    main()
