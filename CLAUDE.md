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
- **Leaks come in batch failure waves (an AGE spike whose age drifts by
  batch)** -- supersedes the earlier "vintage effect" reading. ~82% of
  2019-2026 leaks happen at ages 61-67. Each installation batch fails in
  a ~1-year window (sd of failure age ~0.45 y); ~75% of a batch laid
  1957+ fails in it (~45% for 1954-55), and the survivors drop back to
  ~0.01 leaks/year. The failure age gets younger for newer batches,
  ~0.26 y per lay year (~65 for 1955, 63.5 for 1959, 62.5 for 1963,
  ~62 for 1964-65). Within a batch the leak date tracks the exact lay
  date one-for-one, so use exact (decimal) ages, not year differences.
  Consequence for 2027: the 1964 and 1965 batches (~4,900 unleaked
  pipes) are mid-spike and most will leak in 2027-28 unless replaced on
  1 Jan 2027/2028. The 1971 and 1982 batches leak at an elevated ~5%/y
  since 2022 but no spike has been observed for them (yet). Background
  hazard for all groups rose around 2022-23. Model: 
  `scripts/batch_wave_hazard.py`.
- **Leaks trigger other leaks (sporadic / cascading / clustered)**: 1,118
  timestamps (date + minute) are shared by 2-12 leaks -- 4,112 leaks, 29%
  of all (random collisions would give ~24 pairs; only 17 leaks have time
  00:00, so not a missing-time artefact). Read as: a source leak plus the
  leaks it triggers at the same instant (~3,000 triggered leaks, ~21%).
  For a non-poly pipe at straight-line distance d from a leaking pipe, the
  chance of leaking at the same instant is ~1.3/1000 at d < 50 m, 0.5 at
  200-300 m, 0.1 at 600-800 m, 0.02 beyond 1.2 km; pipes connected to the
  source through the network are ~2-4x more likely than unconnected pipes at
  the same straight-line distance (cascading on top of clustering). Network =
  pipes sharing an endpoint within 0.5 m (coordinates are rounded ~0.1 m);
  largest connected piece 34,217 pipes. Our risk models treat every leak as
  independent, so they cannot credit a plan for cutting off triggered leaks.
  **Modelled (instructor's lecture: sporadic / cascading / clustered)** --
  `scripts/leak_events.py` splits leaks into 10,001 sporadic, 1,118 event
  sources, 2,994 victims; `pipeplan.contagion.fit_trigger_model`: P(victim) =
  distance kernel (straight line x connected-or-not; hop count adds little)
  x victim factor x source factor x same-batch bonus. Victims are mostly the
  SOURCE'S OWN BATCH (79% same lay year vs 5% of nearby pipes; same lay group
  x3.8), copper and 1954-59 pipes are prone, pre-1954 / 1966+ pipes rarely
  involved; 1964-65 sources x3.1. In-sample 0.27 victims per primary vs
  0.25 observed. Victims are the batch's spike SURVIVORS (low own hazard), so
  triggered leaks are extra leaks, not just earlier ones.
  **But it does not explain the public-score gap**: in the contagion world
  (`pipeplan.contagion_sim`, primary-only hazard) the $74.83M plan scores
  $60.7M (sd 5.8, max of 40 heavy-tailed draws $70.4M). Only ~317 of ~3,900
  future leaks are triggered (~$3.6M), because the plan replaces the spiking
  batches' neighbourhoods in the first years -- so a contagion-aware planner
  can gain at most ~$3.6M, realistically ~$1M.
- **Leak cost differs by batch**: 2019-2026 leaks of the 1964-65 batches cost
  0.83x the batch-blind cost GLM (0.69x without the top 0.5%), 1966+ 0.67x
  (0.29x), <1954 1.22x. A cost GLM with batch group (`cost_model.py`,
  `glm_lay`) only partly fixes it out of sample (LOYO 1964-65 1.17 -> 1.12,
  1966+ 1.52 -> 1.21) and has worse deviance -- not adopted.
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

- **Scoring formula (2026-11-09)**: the area term is the *corrected*
  reading, `surface_rate * area^0.85` (not `(surface_rate * area)^0.85`
  as printed above and in the original PDF). This costs `surface_rate^0.15`
  times more per circle (~1.8x on structure, ~1.5x on road): the area term
  of the batchwave plan goes from $49M to $84M, $60M of it on water and
  structure circles. Build and score plans with `--formula corrected`
  (`pipeplan.simulator.COST_FORMULA` still defaults to `"original"`).
  Corrected-formula submission: `submission/submission_batchwave_corrected.csv`.
  **Confirmed against the evaluator** with a diagnostic upload
  (`scripts/probe_scoring.py`, `submission_probe_area.csv`: last circles
  enlarged so only the area term changes): public score moved -$6.10M vs
  -$5.99M predicted (corrected) and -$3.21M (original).
- **Public vs final score**: the final grade uses a separate private future
  drawn iid from the same distribution; the team picks which upload counts
  and uploads are unlimited. Choose plans by offline expected score across
  risk worlds, never by the best of several similar public scores. Probes
  that measure fixed facts (cost rules, which pipes are polyurethane today)
  are fine; probes that measure which pipes leak in the public future are not
  (that does not transfer to the private future).
- **Pre-1954 "iron" pipes are really iron (probe 2,
  `submission_probe_oldpoly.csv`)**: hypothesis was that many of the
  ~15,000 pre-1954 pipes were already polyurethane from leaks before the data
  starts. 90 single-old-pipe circles enlarged by ~$67k each: public score
  moved -$5.45M vs -$5.08M predicted if none are polyurethane (~-$3.3M if
  35% were), so essentially all are iron. Treat `pipes.csv` iron as iron.
- **Probe 3 (`scripts/probe_swap_years.py`, 2027 and 2028 circle blocks
  swapped)**: public -$19.48M vs batchwave -$42.4M (90% -55.5 to -31.2), w060
  -$21.1M. Scaling the 1964-65 spike by f fits at f ~0.36 (probe noise allows
  ~0.2-0.5): `scripts/calibrate_spike_from_probe.py`,
  `leak_risk_batchwave_s035.csv` etc. Plan built on it:
  `submission_batchwave_s035_corrected_leakaware.csv` (+$0.3-0.7M offline in
  the worlds the probe supports).
- **Probe 4 (`scripts/probe_length_charge.py`, 505 single-pipe units for
  left-out copper/brass appended at the end)**: public -$5.99M vs -$4.5 to
  -$5.2M predicted with $200/m and ~-$2.7M without: the length charge is
  real (cost rule fully verified). Slightly larger drop -> fewer of those
  pipes leaked first than modelled.
- **What the level implies**: with NO leaks the uploaded plan scores exactly
  $76.23M (spend $173.77M); public $74.83M means leak repair cost exceeded the
  replacement cost leaks saved by only ~$1.4M (model: ~$16M) -- like ~10% of
  modelled leak rates. Yet probe 3 shows the mid-spike batches (1964 wrought,
  1965 gray/cast) DO leak heavily in 2027 if not replaced. Working hypothesis:
  those leak as modelled, almost everything else (1965 wrought, background,
  1966+ cast, copper/brass) leaks far less.
- **The 2027+ leak process has ~30% of the modelled leak rate
  (`scripts/world_check.py`, `data/processed/world_check.csv`)**: 11
  interpretable worlds scored against all 8 public numbers from the one public
  future (level of the $74.83M plan + 7 upload/probe differences, paired
  heavy-tailed MC). Only "every leak hazard x0.3" fits all (Mahalanobis^2 6.9
  on 8 numbers, every |z| < 1.9: swap probe -17.0 vs -19.5, length probe
  -5.98 vs -5.99, level 70.0 vs 74.8); batchwave 31.8, spike x0.35 24.0,
  background-only or cost-only changes 22-72. Risk files:
  `data/processed/worlds/leak_risk_<world>.csv`. Plan with
  `worlds/leak_risk_all_x0.3.csv` (runner-ups: `w1965_off_bg_x0.3_cost_x0.7`,
  `w1965_off_bg_x0.3`). Consequence: with leaks this rare, spend is the main
  lever -- the uploaded plan's cost if nothing leaks is $173.77M (score
  ceiling $76.23M), while 6 of 7 other teams score $77.6-80.4M.
- **Unexplained (before probes 3-4)**: public scores sit ~$16M above every risk model's
  prediction (0 of 200 heavy-tailed futures reach them). Cost rules and the
  network state are verified, so the gap is in future leaks themselves
  (fewer and/or cheaper than modelled).

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
- **Circle design, set cover (2026-11-09)**: `scripts/build_plan.py --construct
  setcover` (`pipeplan.setcover`) replaces pairwise merging with ~435k candidate
  circles (each target pipe + its 1..25 nearest targets, again with same-surface
  neighbours only, plus every circle of `--sc-seed-plans`) and an exact HiGHS set
  cover on fixed cost (`100 + area term + swallowed copper/brass`). Area+fee cost
  $84.71M -> $83.35M; ordering unchanged (leak cost +$0.06M). Paired heavy-tail
  MC vs `x03_leakaware`: +$0.94-1.05M in all_x0.3, both w1965_off worlds and
  batchwave_s035 (wins 83-97%). File: `submission_x03_leakaware_setcover.csv`.
  Bigger pool (`--sc-k 40 --sc-same-k 40 --sc-rmax 200`, seeded with both
  plans; ~660k candidates) saves another ~$0.3M: `submission_x03_leakaware_setcover40.csv`,
  +$1.09-1.29M vs `x03_leakaware` in the same 4 worlds (wins 87-97%). Current best
  offline plan (not yet uploaded as of 2026-11-09). Paired results:
  `data/processed/compare_setcover.csv`.
  Timing is ignored when choosing circles (ordering still is time-aware).
  **There is no timing headroom left**: one circle per pipe (maximal ordering
  freedom, leak-aware LP order) still has expected leak cost $19.30M vs $19.25M
  for setcover40 (all_x0.3 world) -- leak cost is set by the $10M/yr pace and the
  hazard, not by grouping. `--sc-lam` (time-aware penalty, `setcover.delay_penalty`)
  changed nothing at lam 0.03-0.3; leave it off.
  **Radius cushion**: `--sc-cushion 0.01 --sc-min-margin 0.005` (default 0.05 /
  0.04) saves ~$0.27M spend with every iron pipe still >= 5 mm inside (coordinates
  and circles are both in whole cm, so this is far beyond float error).
  `submission_x03_leakaware_setcover40_c1.csv` = setcover40 with that cushion:
  +$1.34-1.55M vs `x03_leakaware` (+$0.24-0.27M vs setcover40) in all 4 worlds,
  wins 89-98%; exact integer-cm check: 0 of 26,245 iron pipes uncovered. **Current
  best offline plan (not yet uploaded, 2026-11-09).** If an upload is ever
  rejected for coverage, fall back to `submission_x03_leakaware_setcover40.csv`.
- **Circle design (original)**: heuristic spatial clustering (e.g. DBSCAN / greedy
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

## Shared outputs (schemas)

- `data/processed/leak_risk_*.csv` (all risk models): `Pipe ID` (str),
  `cost_if_leak` ($, expected cost given a leak), `p_leak_annual`
  (10-year-average yearly probability, for the constant-hazard planner),
  `p_2027` .. `p_2051` (P(leak in that calendar year | not leaked by its
  start)). One row per pipe still non-polyurethane in 2027 (27,248).
  `leak_risk_batchwave.csv` / `leak_risk_batchwave_extrap.csv` come from
  `scripts/batch_wave_hazard.py` (`_extrap` = drifting spike also applied
  to the 1966-1985 batches).
- Planner: `scripts/build_plan.py --yearly` makes the planner use the
  `p_YYYY` columns (time-aware expected score, Monte Carlo, and an LP
  year-assignment ordering in `pipeplan.ordering.lp_order`); without
  `--yearly` it behaves exactly as before (constant `p_leak_annual`).
  `--leak-aware` (needs `--yearly`) makes the ordering (Smith + LP) and
  `expected_score` credit pipes that leak before their circle runs (no
  $200/m for them, all-leaked circles skipped): the planner's estimate then
  matches the Monte Carlo (it over-counted spend by ~$31M). Paired
  heavy-tail MC, corrected formula: +$0.9M (batchwave risk), +$2.2M (w060),
  +$1.0M (batchwave_extrap) vs the same circles ordered without it, positive
  in 74-90% of futures. Circle construction is not leak-aware yet.
  `expected.last_year_no_leaks` in the plan summary is the finishing year if
  nothing leaks (worst case for the 25-year rule; build_plan asserts < 25).
- `data/processed/plan_backtest_batchwave.csv`
  (`scripts/plan_backtest_batchwave.py`): one row per cutoff x
  `risk_model` (`w060` | `batchwave`) x `planner` (`const` | `yearly`);
  `score`/`spend`/`leak_cost` ($) against the real leaks after the
  cutoff, `score_expected_cost`/`leak_cost_expected_cost` (same leaks
  valued at the cost model's expected cost), `n_leaks`, `vs_w060*`
  (difference from the w060 pipeline at the same cutoff, $).
- `data/processed/cost_model_holdout.csv` (`scripts/cost_model.py`): one
  row per cutoff (2021-2025) x cost `model` (`glm_base` = current Gamma GLM,
  `glm_inter` + material x surface + age, `glm_full` + spatial kNN residual,
  `lgbm`); `gamma_dev`, `spearman` (overall and `_road`/`_structure`),
  `pred_over_actual_*` (predicted $ / actual $ on leaks after the cutoff),
  `top10_share_actual_usd`. Result: none of the engineered models beats
  `glm_base`, because leak cost beyond surface is mostly unpredictable
  heavy-tailed noise (top 100 of 14,113 leaks = 40% of all leak $).
- `data/processed/cross_evaluation_batchwave.csv`
  (`scripts/cross_evaluate_plans.py`): one row per plan x risk world;
  Monte Carlo mean/sd score, leak cost, spend, number of leaks ($).
- `data/processed/leak_risk_batchwave_mix50.csv`
  (`scripts/mix_risk_scenarios.py`): same schema as every leak_risk file;
  50/50 mixture of `batchwave` (1966-85 batches keep a flat ~5%/y) and
  `batchwave_extrap` (they spike, 1971 batch ~2031-32, 1982 ~2039-40),
  mixed on survival curves. Finding: the 1971/1982 cast-iron rate has been
  flat since the 2022 jump with no age gradient inside the 1971 batch, but
  the 1964-65 cast-iron batches showed the same plateau right before their
  spike, so the data cannot decide. It is worth little either way:
  plans built for no-spike / spike / 50-50 differ by <= $0.7M in every
  world (paired heavy-tail MC, corrected formula, leak-aware ordering).
- `data/processed/leak_events.csv` (`scripts/leak_events.py`): one row per
  2019-2026 leak: `Pipe ID`, `ts` (date + minute), `event_id` (-1 = single
  leak), `role` (`sporadic` | `source` | `victim`), `source_pipe`,
  `d_euclid_m` / `d_net_m` / `d_hops` (victim to source; inf = not connected).
  `data/processed/trigger_kernel.csv`: co-leak probability by straight-line
  bin x hop bin. `data/processed/trigger_model.json`: fitted
  `pipeplan.contagion.TriggerModel` (kernel by distance bin x connected, log
  multipliers for victim/source material, surface, lay group, same
  material / lay year / lay group); load with
  `pipeplan.contagion_sim.load_trigger_model()`.
- `data/processed/leak_risk_batchwave_primary.csv`
  (`batch_wave_hazard.py --primary-only`): leak_risk schema, PRIMARY leaks
  only (victims censored) -- use only together with the trigger model
  (`pipeplan.contagion_sim.monte_carlo_events`), otherwise leaks are
  undercounted.
- `data/processed/cost_model_loyo.csv` (`scripts/cost_model.py --loyo`):
  leave-one-year-out (2019-2026) cost-level check, one row per `test_year`
  x `model`; `gamma_dev`, `pred_usd`/`actual_usd` and `pred_<surface>` /
  `actual_<surface>` ($). Result: the current Gamma GLM (`glm_base`) is
  correctly calibrated (pooled pred/actual 0.98; road 0.98, structure
  0.97); single years range 0.54 (2021) to 1.23. The 1.1-1.65x $
  overprediction in the rolling backtests is an artefact of overlapping
  test windows (all contain 2026, none contain the 2021 outliers), not a
  level error -- do not shrink the cost level. Lognormal alternatives
  misallocate across surfaces (road 1.26, structure 0.75).
- `data/processed/backtest_leak_counts.csv` (`scripts/backtest_leak_counts.py`):
  one row per cutoff (2021-2025) x `model` (`hazard`, `lgbm`, `ens_w050`,
  `ens_w060`, `batchwave`, `batchwave_mat` = `batch_wave_hazard.py
  --material-spike`, per-material spike offsets; fixes the gray-iron count
  but backtests slightly worse on logloss/top-10% $, so not the default) x
  `year` (`h` = year -
  cutoff) x `view` (`one_step` = among pipes alive at the start of the
  year; `cumulative` = all pipes at risk at the cutoff, through that year)
  x `group`/`level` (`all`, `surface`, `material`, `lay` bucket);
  `pred_n`/`actual_n` (leaks), `pred_usd`/`actual_usd` ($, cost GLM),
  `ratio_n`, `ratio_usd` (pred/actual). Result: batchwave gets the leak
  count right (cumulative 0.97-1.07); hazard/lgbm/ensembles predict 25%
  too few leaks one year out and only ~31% of them five years out
  (they miss the 1960-65 batch waves).
- `data/processed/cross_evaluation_heavytail.csv` (`scripts/cross_evaluate_plans.py
  --heavy-tail`): same columns, but each leak's size is drawn from the real
  2019-2026 cost distribution (`pipeplan.simulator.empirical_cost_ratios`),
  plus `mc_score_p05`/`mc_score_p95` and `diff_vs_first_mean`/`_sd` (paired
  difference to the first plan, same futures). Realistic score sd is
  ~$5M per plan and ~$4M for the difference between two plans; the
  default (mean-cost) Monte Carlo sd of ~$1.2M is far too small.

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
