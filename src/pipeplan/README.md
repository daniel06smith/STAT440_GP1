# pipeplan - scoring simulator, circle construction and ordering

Plan step 5 of the group plan ("identify the areas and ordering of work units"),
built on the `qusai` branch. Everything is plain numpy / pandas / scipy.

## Quick start

```bash
python tests/test_pipeplan.py        # 12 unit tests (hand-checkable toy networks)
python scripts/build_plan.py         # whole pipeline, ~90 s -> submission/submission_v1.csv
python scripts/sensitivity.py        # optional, ~5-8 min -> data/processed/sensitivity_construction.csv
```

Notebooks (jupytext "percent" `.py` with a paired `.ipynb`, same convention as
`notebooks/02_eda_python.py`): read them in this order.

| Notebook | What it explains |
|---|---|
| `notebooks/04_scoring_simulator.py` | the professor's rules as code, worked cost example, tests, naive baseline |
| `notebooks/05_circle_construction.py` | how pipes are grouped into circles, and why |
| `notebooks/06_ordering_and_evaluation.py` | the order of the circles, final checks, the submission file |

## Files

| File | Role |
|---|---|
| `world.py` | loads the pipes + leak-risk table into flat arrays; "pipe wholly inside circle" geometry |
| `geometry.py` | convex hull, smallest enclosing circle |
| `simulator.py` | `simulate` (exact rules, one future), `monte_carlo`, `expected_score` (fast) |
| `construct.py` | `Builder`: bottom-up merging of circles with the true cost + a delay penalty |
| `ordering.py` | iterated Smith's rule, baseline orderings, dropping skipped circles |
| `pipeline.py` | glue: build the plan, write/read the submission, independent coverage check |

## Using another leak model, or the other cost formula

```bash
python scripts/build_plan.py --tag hazard --formula corrected     --risk-csv path/to/priority_all_hazard.csv --id-col id --p-col p --cost-col cost_if_leak --horizon-years 5
```

* `--horizon-years` says what the probability column means: 1 = per year (Cheryl), 8 = over
  2019-2026 (Kenson's `priority_all.csv` / `_stepwise.csv`), 5 = next 5 years (Kenson's `_hazard.csv`).
  It is converted to a yearly probability `1-(1-p)^(1/years)`; the simulator then assumes that
  chance is the same every year (so Kenson's rising hazard is averaged over the window).
* `--formula original` = `(surface_rate x area)^0.85`. **This is what the website's data page
  states (checked 28 Oct 2026) and what every submission uses.** `corrected` =
  `surface_rate x area^0.85` is the reading Wendy's notebook uses; under it our simulator
  reproduces her reported replacement cost ($187,733,311.63) to the cent, which also validates
  our cost logic. Her plan's cost figures are therefore too high for the real formula.
* Output files are named by `--tag`; the default (`v1`) reproduces the committed plan exactly.

## Inputs and where they come from

* `data/raw/pipes.csv`, `data/raw/train.csv` - the professor's data (identical to `data.zip`; MD5 checked).
* `data/processed/leak_risk_cheryl.csv` - **Cheryl's** per-pipe leak model output
  (`p_leak_annual` = yearly leak probability, `cost_if_leak` = expected cost of a leak)
  copied from `priority_all.csv` on branch `Cheryl` (commit `f7472c3`). We use her model rather than
  re-fitting so that the whole group's work feeds into one plan.

## Decisions made (and why) - change them in one place if the group disagrees

| Decision | Where | Why |
|---|---|---|
| The 8 pipes with blank material are treated as wrought iron and must be covered | `world.build_world` | one uncovered iron pipe = rejected submission. Same choice as Daniel, Cheryl, Kenson. (The `julia-prediction` notebook drops them; we keep them to be safe.) |
| A pipe is "inside" iff both end points are inside the closed circle | `World.members` | project text: "wholly within the circle" |
| Radius is rounded **up** to the cent with a ~5 cm cushion (every iron pipe ends up >= 4 cm inside some circle) | `Builder.circles` | cannot lose an iron pipe to rounding or tolerance in the website's checker (costs ~0.4% of score) |
| Copper/brass pipes are optional targets, taken if they pay for themselves within 12 years | `construct.choose_targets` | they leak too and we pay for those leaks; worth ~$8M in the model |
| Leaks are exponential with the yearly probability from Cheryl's model | `simulator.sample_future` | `P(leak within y years) = 1-(1-p)^y`, consistent with her `p_leak_annual` definition |
| Leak cost = her model's mean cost | `World.leak_cost` | real costs are heavy-tailed; Monte Carlo spread understates the real noise |

## Limitations

See the last cell of notebooks 04, 05 and 06. The most important: nothing here has been
checked against the real website yet, and every dollar figure depends on the leak model.
