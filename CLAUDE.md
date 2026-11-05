# STAT 440 Group Project 1 -- Star Kingdom pipe replacement

## The task (see `Project_Description.md` for the verbatim spec)

Star Kingdom has ~43,000 gas pipes (`data/raw/pipes.csv`) made of iron
(`wrought iron` / `cast iron` / `gray iron`), `copper`, `brass`, or
`polyurethane`. All iron pipes must be replaced with polyurethane within a
25-year project window (2027-2051), under a budget of $10,000,000/year
($250M total, non-rolling except that unfinished-year work carries into the
next year's budget). We choose:

- **Work units**: circles `"(x, y)",r` (GPS metres). Executing a work unit
  replaces every non-polyurethane pipe *wholly inside* the circle with
  polyurethane.
- **Order** of execution across the whole 25-year window.

**Cost of a work unit**: `100 + (surface_rate * area)^0.85 + 200 * (metres
of non-polyurethane pipe wholly inside)`, where `surface_rate` is the *sum*
of $/m² rates for every distinct surface type appearing among **all**
pipes wholly inside the circle (grass $3, farm $7, swamp $12, road $17,
water $32, structure $50) -- not just the replaced ones, and not
proportional to how much of that surface there is.

**Score** = `250,000,000 - (sum of executed work-unit costs) - (sum of
leak repair costs incurred 2027-2051)`. The spec is rejected outright if
any iron pipe ends up unreplaced, or if the plan can't finish in 25 years.

### Loss function to minimize

Since the $250M budget term is fixed, **maximizing score <=> minimizing
total cost**:

```
L = sum_over_executed_work_units( 100 + (surface_rate_c * area_c)^0.85 + 200 * length_non_poly_in_c )
  + sum_over_all_leaks_in_2027..2051( repair_cost )
```

The first term is (almost) fixed by which pipes must be swept up and how
circles are drawn; the second term is *stochastic* future leak cost we
don't observe directly -- we only have `data/raw/train.csv` (leaks from
2019-2026) to estimate leak hazard and cost from. So in practice we
minimize **expected** total cost:

```
E[L] = (deterministic replacement/earthwork cost given circle design)
     + sum_over_not-yet-replaced_pipes_p( P(p leaks before its circle executes) * E[cost | p leaks] )
```

This decomposes the whole problem into two coupled subproblems:

1. **Circle design** (geometry): which pipes get grouped into which
   circles -- controls the earthwork/area term and how much copper/brass
   gets opportunistically caught alongside iron.
2. **Execution order / schedule**: which circles run first, subject to the
   $10M/year pacing -- controls expected leak exposure, i.e. how long each
   pipe sits unreplaced and at risk.

A key EDA finding (see below) is that proactively replacing a pipe is
almost always cheaper in expectation than leaving it to possibly leak, so
subproblem (2) mostly means: **replace the highest `P(leak) x E[cost |
leak]` pipes as early as the budget schedule allows**, not "replace
cheapest circles first."

## Data

- `data/raw/pipes.csv` (43,039 rows): `Pipe ID`, `Lay date`, `Material`,
  `Surface`, `GPS x1/y1/x2/y2` (pipe endpoints, metres). `Material` records
  the pipe's *original* material even for pipes already replaced via a
  leak repair before 2019 or during 2019-2026 -- see below.
- `data/raw/train.csv` (14,113 rows, "costs.csv" in the spec text): leak
  history 2019-2026 -- `Pipe ID`, `Date`, `Time`, `Cost`. Every pipe here
  leaked exactly once (no repeats) and was repaired with polyurethane on
  the spot.
- `data/processed/merged.csv`: `pipes.csv` joined to `train.csv` (existing
  R-side artifact, one row per historical leak with full pipe attributes).

## EDA status

Two EDA notebooks exist and should both be treated as living references,
not one-off scratch work:

- `notebooks/01_eda.qmd` (R) -- first-pass material/surface distribution
  checks.
- `Visualization.qmd` (R) -- spatial maps of surface type and pipe
  material.
- `notebooks/02_eda_python.ipynb` (Python, paired with
  `notebooks/02_eda_python.py` via jupytext for readable diffs) -- the
  main EDA supporting the plan above. Re-run after any `pipes.csv` /
  `train.csv` change with:
  ```
  uv run jupytext --to notebook notebooks/02_eda_python.py
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_python.ipynb
  ```

### Findings that drive the plan

- **"Iron" = `wrought iron` (24,562) + `cast iron` (10,630) + `gray iron`
  (3,263)** = 38,455 pipes (89% of the network). `copper` (1,613) and
  `brass` (1,285) are *not* mandated for replacement.
- **32.8% of the network (14,113 pipes) is already effectively
  polyurethane** as of 2027 (leaked in 2019-2026 and was repaired), even
  though `pipes.csv` still lists the original material for those rows.
  Effective non-poly pipes actually needing a work unit: **27,248**
  (26,237 iron + 696 copper + 307 brass), totalling **471,777 m**.
  Always compute an `effective_material_2027` column (poly if `Pipe ID` is
  in `train.csv` OR `Material == polyurethane`) before doing anything
  downstream -- don't use raw `Material` directly.
- **Risk (probability) and cost-given-leak both matter, and neither tracks
  excavation cost directly**:
  - Leaks-per-km: `brass` 68.8 (highest of all materials, though not
    mandated), `wrought iron` 29.6, `copper` 14.8, `cast iron` 11.7,
    `gray iron` 8.7 (least risky iron alloy). By surface: `farmland` 27.2,
    `grassland` 20.8, `swamp` 18.5, `water` 17.9, `road` 14.4,
    `structure` 13.5 (safest despite the highest per-m² excavation rate).
  - **Cost given a leak is wildly surface-dependent**: `road` leaks cost a
    median $71,592 / mean $232,300 -- roughly 10-30x every other surface
    (which run ~$2,000-12,000 median). `road` is only mid-pack on
    frequency, so this is a severity effect easy to miss from counts
    alone. By material, `brass` and `gray iron` have the highest per-leak
    cost (median ~$7,500-9,400) despite `wrought iron` leaking most often.
  - Net implication: prioritize by `P(leak) x E[cost | leak]`, and treat
    `road`-surface pipes as high priority regardless of material/hazard.
- **Letting a pipe leak instead of proactively replacing it is a net loss
  in expectation almost everywhere.** Proactive replacement saves ~$200/m
  x avg length (~17.3 m) = ~$3,460 in linear cost. That's smaller than the
  *mean* leak cost on every surface except farmland/grassland/water, and
  tiny next to the median leak cost on `road` ($71,592) or `structure`
  ($11,850). So "replace early" dominates "wait and get a free
  replacement" for anything except the very lowest-risk, lowest-cost
  pipes.
- **Age effect is non-monotonic, likely an installation-era/vintage
  effect**: pipes laid ~1959-1979 (40-60 years old as of 2019) leak at
  467/1000, roughly double the 20-40y (212/1000) and 60-80y (210/1000)
  cohorts. All pipes were laid 1945-1985. Don't assume linear
  age-hazard -- model lay-year cohort explicitly, and remember the
  project window pushes ages 20 years further (up to ~105y by 2051),
  beyond the observed training range, so extrapolation behavior matters.
- **Spatial clustering is tight**: median nearest-neighbor distance
  between to-be-replaced pipe midpoints is 7.9 m (75th pct 13.6 m, long
  thin tail to 131 m) -- small/moderate-radius circles can plausibly sweep
  up many pipes each.
- **Budget looks comfortable if circles are drawn sensibly**: the
  unavoidable linear-cost floor (`$200/m x 471,777 m`, independent of
  circle geometry) is $94.4M of the $250M budget (38%), leaving ~$155.6M
  for the base+area^0.85 terms combined. The $100/circle base fee is
  trivial even at one circle per pipe (~$2.7M worst case). Real budget
  risk lives in circle area/surface-mix, not fee count or the linear
  floor.
- **Data quality**: 8 pipes (all `water` surface) have unknown `Material`
  and never appear in `train.csv` -- default assumption is to treat them
  as iron (safe: can't cause a spec rejection for a missed pipe). 15
  `wrought iron` pipes have missing `Lay date` -- material is known so
  they still must be replaced; only age-based risk scoring is affected
  (impute, e.g. cohort median lay date).

## Decisions made

- **Copper/brass**: opportunistically sweep them into iron-targeting
  circles when cheap to do so (not required, but `brass` is the
  single riskiest material by leak frequency).
- **Tooling**: the group is mixed R/Python (some members prefer each).
  Both are fine to use side by side -- see "Cross-language workflow"
  below for the interchange convention.
- **Risk model**: fit a formal statistical hazard/rate model (e.g.
  Poisson or Cox-style regression on material x surface x lay-year
  cohort x location) rather than relying on the empirical leaks-per-km
  lookup tables alone. Reasons: it extrapolates more defensibly to ages
  beyond the observed 20-80y training range (project window reaches
  ~105y by 2051) and handles sparse strata combinations better. The
  empirical tables from `notebooks/02_eda_python.ipynb` should still be
  used as a validation baseline the fitted model is expected to
  reproduce.
- **Circle design**: heuristic spatial clustering (e.g. DBSCAN / greedy
  circle packing seeded from the ~8-14m nearest-neighbor spacing found
  in EDA, ordered by risk density) with local-search refinement on
  radius/center, rather than an exact formal optimization. Reason: the
  `area^0.85` term is nonlinear/non-convex, and the problem is at
  27,248-pipe scale, making an exact MILP formulation impractical for
  the project timeline.

## Cross-language workflow (R + Python)

Group members work in whichever of R or Python they prefer. To keep this
workable:

- **All shared intermediate/final outputs are plain files** (CSV or
  Parquet) under `data/processed/` (or a new `outputs/` dir for
  model-specific artifacts like risk scores or work-unit specs), never
  language-native serialization (no `.rds`, no `pickle`) -- both R
  (`readr`/`arrow`) and Python (`pandas`) can read either format directly.
- **Document every shared file's schema** (column names/types/units)
  either in this file or a header comment/markdown cell in the notebook
  that produced it, so the other language's users don't have to
  reverse-engineer it.
- Notebooks stay paired with their source language: `.qmd` for R (Quarto),
  `.ipynb` (+ a jupytext `.py` twin for diffs) for Python. Don't
  transliterate one language's notebook into the other -- keep one
  canonical notebook per analysis and cite/link it from anywhere it's
  reused.
- The final work-unit specification CSV (the deliverable format from
  `Project_Description.md`: `"(x, y)",r` per line) is the one artifact
  everyone must be able to produce/validate identically regardless of
  language -- write a small validator (checks: every iron pipe wholly
  covered by >=1 circle, budget/year feasibility simulation) and keep it
  in sync in both languages, or pick one language as the source of truth
  for validation and document that choice here once made.

## Python environment

Managed with `uv` (`pyproject.toml` + `uv.lock`). Setup:

```
uv venv
uv sync
```

`.env` holds `DATA_RAW_DIR` / `DATA_PROCESSED_DIR` (relative paths),
loaded via `python-dotenv` -- lets notebooks run correctly regardless of
whether they're launched from the repo root or `notebooks/`.

Registered Jupyter kernel: `stat440-gp1` ("STAT440_GP1 (uv)").
