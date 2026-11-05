# %% [markdown]
# # 06 - Ordering the work units, evaluating the plan, writing the submission file
#
# **Author of this notebook:** Qusai (qusai branch). Plan step 5, part 2 of 2.
# Input: `data/processed/circles_unordered_v1.csv` from notebook 05.
#
# ## Why does order matter?
# * Money arrives at $10M a year and units are dug **strictly in file order**, so a
#   circle's position decides **which year** it is dug.
# * A pipe still in the ground in year *y* has had *y* years to leak:
#   expected loss = `leak_cost x (1 - (1 - p)^y)`.
# * Every pipe is replaced exactly once whatever the order, so total replacement spend is
#   (almost) fixed - **the order only changes the leak bill.**
#
# ## The ordering rule (Smith's rule, iterated)
# For a single worker doing jobs one after another, the order that minimises
# `sum(weight x finish time)` is *largest weight / duration first*. Here
# * duration = what the circle costs (money is the scarce resource, so cost plays the role of time),
# * weight = expected dollars lost **per year** if the circle's pipes stay in the ground
#   (`sum of K x hazard` over the pipes it replaces).
#
# So: dig first the circles that remove the most expected leak-loss per dollar spent.
# The weight shrinks with time (pipes that survive are, by definition, still intact, and
# the survival probability decays), so we iterate: schedule -> estimate each circle's
# year -> recompute weights at that year -> re-sort (8 rounds, damped).

# %%
from pathlib import Path
import sys

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeplan.world import load_world
from pipeplan.simulator import prepare, expected_score, monte_carlo
from pipeplan.ordering import smith_order, baseline_orders, permute, drop_redundant
from pipeplan.pipeline import (circle_table, write_submission, read_submission,
                               independent_coverage_check)

pd.set_option("display.width", 160)
w = load_world()
circles = pd.read_csv(ROOT / "data/processed/circles_unordered_v1.csv").to_numpy()
prep0 = prepare(w, circles)
print(len(circles), "circles | covers all iron:", prep0.covers_all_iron)

# %% [markdown]
# ## 1. Does the ordering rule beat simpler orderings?
# Same circles, six orderings. Scored two ways: the fast `expected_score`, and the
# Monte Carlo average over 25 random futures (event-exact rules, notebook 04).
# * *random* - no thought at all
# * *by x coordinate* - sweep west to east (what a "work through the map" plan would do)
# * *by max leak probability* - riskiest single pipe first
# * *by total loss rate* - biggest expected annual loss first, ignoring cost
# * *static Smith* - loss rate per dollar at time 0 (one pass of the rule)
# * *iterated Smith* - the final rule (below)

# %%
best, log = smith_order(w, prep0, iters=8, verbose=True)

# %%
rows = []
orders = baseline_orders(w, prep0)
variants = {k: permute(prep0, o) for k, o in orders.items()}
variants["iterated Smith (final)"] = best
for name, p in variants.items():
    e = expected_score(w, p)
    m = monte_carlo(w, p, n_draws=25, seed=11)
    rows.append({"ordering": name,
                 "expected_score_M": e["score"] / 1e6, "expected_leak_M": e["leak_cost"] / 1e6,
                 "MC_score_M": m["score_mean"] / 1e6, "MC_score_sd_M": m["score_sd"] / 1e6})
cmp = pd.DataFrame(rows).round(2)
cmp

# %%
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(cmp["ordering"], cmp["MC_score_M"], color=["#bbb"] * (len(cmp) - 1) + ["#2b7bba"])
ax.set_xlabel("Monte Carlo mean score ($ million)")
ax.invert_yaxis()
plt.tight_layout(); plt.show()

# %% [markdown]
# The ordering is worth tens of millions of dollars: a random order scores about $18M in
# Monte Carlo, against about $96M for the Smith order. Both scoring methods rank the six
# orderings the same way, so optimising the fast score is safe. (We have not proven the
# Smith order optimal - only that it beats these alternatives.)

# %% [markdown]
# ### Convergence of the iteration
# Iteration 0 is plain Smith (weights at time 0); later iterations re-weight each circle at
# the year it is scheduled for. It gains ~$0.4M and flattens out.

# %%
pd.DataFrame(log).assign(score_M=lambda d: d.score / 1e6, leak_M=lambda d: d.leak / 1e6)[["iter", "score_M", "leak_M", "last_year"]].round(3)

# %% [markdown]
# ## 2. The final plan

# %%
# Remove circles that would be skipped (all their pipes are replaced earlier). One safety
# guard: a skipped circle is kept if it is the only one containing some iron pipe with at
# least 4 cm to spare (otherwise that pipe would hang on a circle that holds it by a few mm).
final = drop_redundant(w, best)
print("circles:", prep0.n, "->", final.n, "after dropping circles whose pipes are all replaced earlier")
table = circle_table(w, final)

by_year = table.groupby("year").agg(
    circles=("order", "size"), pipes=("pipes_replaced", "sum"), iron=("iron_replaced", "sum"),
    metres=("metres_replaced", "sum"), spend=("exec_cost", "sum"), loss_rate_removed=("loss_rate_replaced", "sum"),
)
by_year["share_of_initial_loss_rate_%"] = 100 * by_year.loss_rate_removed / by_year.loss_rate_removed.sum()
by_year["cum_share_%"] = by_year["share_of_initial_loss_rate_%"].cumsum()
by_year.round(1)

# %% [markdown]
# (A "guard" circle - kept only for the coverage-margin safety rule - has year `NaN`: the real rules skip it for free.)
#
# `loss_rate_removed` is the expected $/year of leaks that disappears when the year's
# circles are dug. Front-loading is clear: the first years remove a large share of the
# total expected loss with a small share of the money.

# %%
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax[0].bar(by_year.index, by_year.spend / 1e6, color="#2b7bba")
ax[0].axhline(10, color="k", ls="--", lw=1)
ax[0].set_ylabel("replacement spend ($ M)"); ax[0].set_title("Spend per year (budget $10M/yr, carry-over allowed)")
ax[1].plot(by_year.index, by_year["cum_share_%"], "o-", color="firebrick")
ax[1].set_ylabel("% of initial expected annual leak loss removed"); ax[1].set_title("Risk removed (cumulative)")
for a in ax:
    a.set_xlabel("year")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Write the submission file and verify it from the file itself
# Format (professor): one line per work unit, `"(x, y)",r`. After writing we **re-read the
# file** with a strict parser and run the independent brute-force coverage check on what
# came back, so what we verify is exactly what would be uploaded.

# %%
sub = ROOT / "submission" / "submission_v1.csv"
write_submission(final.circles, sub)
print(sub.read_text().splitlines()[:3])

back = read_submission(sub)                       # raises if any line is malformed
cov = independent_coverage_check(w, back)
prep_back = prepare(w, back)
print(f"\nlines: {len(back)}")
print(f"iron pipes: {cov['n_required']}, uncovered: {cov['n_uncovered']}, tightest margin {cov['min_margin_m']:.3f} m")
assert cov["n_uncovered"] == 0

# %% [markdown]
# ## 4. Predicted score
# * `expected_score`: fast, conservative (leaks never make a later circle cheaper).
# * `monte_carlo`: event-exact rules over 100 random futures (mean leak cost per leak).

# %%
ex = expected_score(w, prep_back)
mc = monte_carlo(w, prep_back, n_draws=100, seed=2)
res = pd.DataFrame({
    "expected_score (conservative)": [ex["spend"], ex["surplus"], ex["leak_cost"], ex["score"], 2027 + ex["last_year"]],
    "Monte Carlo mean": [mc["spend_mean"], 250e6 - mc["spend_mean"], mc["leak_cost_mean"], mc["score_mean"], 2027 + mc["last_year_mean"]],
    "Monte Carlo sd": [mc["spend_sd"], mc["spend_sd"], mc["leak_cost_sd"], mc["score_sd"], mc["last_year_sd"]],
}, index=["replacement spend $", "surplus $", "leak cost $", "score $", "last work year"])
res.round(1)

# %% [markdown]
# ## 5. How to use / what to check on the website
# 1. Upload `submission/submission_v1.csv`. **First check it is accepted** (not rejected).
#    Acceptance tests our coverage logic and the 25-year feasibility.
# 2. Compare the website score with our range. Expect the real score to differ: the real
#    future leaks are one random draw with heavy-tailed costs, and our leak model is only an
#    estimate. A score *far* outside `Monte Carlo mean +/- 3 sd` (and especially outside
#    the conservative `expected_score`) suggests a rule we mis-read - tell me / the group and
#    we will adjust the simulator.
# 3. The cost of the leaks that really happen is mostly **road** pipes and a few huge
#    leaks; one unlucky draw can move the score by millions.
#
# ## Known limitations
# * The order inside the plan is a heuristic (iterated Smith); we did not prove it optimal.
#   Packing circles tightly against the year boundaries could add a little.
# * Circles are built without knowing the ordering (the delay penalty uses a provisional
#   schedule). A joint optimisation could do better.
# * All numbers rest on Cheryl's leak-risk model and its constant-hazard assumption.
# * Nothing here has been checked against the real evaluator yet.
