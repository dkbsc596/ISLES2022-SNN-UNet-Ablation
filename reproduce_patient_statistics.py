#!/usr/bin/env python3
"""
Reproduce the patient-level statistics of the A0-A3 ablation from
results/publication_results/final_test/patient_level_results.csv :

  * bootstrap 95% CI of every per-model metric mean   (bootstrap_95CI.csv)
  * paired two-sided Wilcoxon signed-rank tests        (statistical_analysis.csv)
  * Holm correction over all 20 tests
  * rank-biserial correlation + bootstrap CI of the mean paired difference
                                                       (effect_sizes.csv)

Output goes to a NEW folder (final_test_reproduced) - frozen results are not modified.
Run from the project root:  python3 reproduce_patient_statistics.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = os.environ.get("PROJECT_ROOT", ".")
FINAL = os.path.join(ROOT, "results/publication_results/final_test")
SRC = os.path.join(FINAL, "patient_level_results.csv")
OUT = os.path.join(ROOT, "results/publication_results/final_test_reproduced")

N_BOOT = 10000
SEED = 42
MODELS = ["A0", "A1", "A2", "A3"]
METRICS = ["dice", "iou", "precision", "recall", "hd95_mm"]
PAIRS = [("A0", "A1"), ("A1", "A2"), ("A2", "A3"), ("A0", "A3")]

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

df = pd.read_csv(SRC)
subject_col = next(c for c in ["subject", "subject_id", "patient", "patient_id"] if c in df.columns)
wide = {m: df.pivot(index=subject_col, columns="model", values=m)[MODELS] for m in METRICS}


def boot_ci(values):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


# ---------- per-model bootstrap CIs ----------
rows = []
for model in MODELS:
    for metric in METRICS:
        v = wide[metric][model].dropna().values
        lo, hi = boot_ci(v)
        rows.append(dict(model=model, metric=metric, n_valid=len(v),
                         mean=v.mean(), ci95_lower=lo, ci95_upper=hi))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "bootstrap_95CI.csv"), index=False)

# ---------- paired tests ----------
res = []
for a, b in PAIRS:
    for metric in METRICS:
        both = wide[metric][[a, b]].dropna()
        x, y = both[a].values, both[b].values
        d = y - x
        nz = d != 0
        W, p = wilcoxon(d[nz], alternative="two-sided", method="approx")
        ranks = rankdata(np.abs(d[nz]))
        w_pos, w_neg = ranks[d[nz] > 0].sum(), ranks[d[nz] < 0].sum()
        r_rb = (w_pos - w_neg) / (w_pos + w_neg)
        lo, hi = boot_ci(d)
        res.append(dict(comparison=f"{a}_vs_{b}", model_A=a, model_B=b, metric=metric,
                        n_paired=len(d), n_nonzero_difference=int(nz.sum()),
                        median_A=np.median(x), median_B=np.median(y),
                        median_difference_B_minus_A=np.median(d),
                        mean_difference_B_minus_A=d.mean(),
                        bootstrap_ci95_lower=lo, bootstrap_ci95_upper=hi,
                        wilcoxon_statistic=W, p_value=p, rank_biserial_effect=r_rb))
res = pd.DataFrame(res)

# ---------- Holm over all tests ----------
p = res["p_value"].values
order = np.argsort(p)
m = len(p)
adj = np.empty(m)
running = 0.0
for rank, i in enumerate(order):
    running = max(running, (m - rank) * p[i])
    adj[i] = min(1.0, running)
res["p_holm"] = adj
res["significant_holm"] = res["p_holm"] < 0.05

res.to_csv(os.path.join(OUT, "effect_sizes.csv"), index=False)
res[["comparison", "metric", "n_paired", "median_A", "median_B",
     "median_difference_B_minus_A", "mean_difference_B_minus_A",
     "wilcoxon_statistic", "p_value", "p_holm", "significant_holm"]] \
    .rename(columns={"n_paired": "n"}) \
    .to_csv(os.path.join(OUT, "statistical_analysis.csv"), index=False)
print(f"Written to {OUT}")

# ---------- compare with the frozen results (if present) ----------
old_path = os.path.join(FINAL, "effect_sizes.csv")
if os.path.exists(old_path):
    old = pd.read_csv(old_path).set_index(["comparison", "metric"])
    new = res.set_index(["comparison", "metric"])
    for col in ["n_paired", "mean_difference_B_minus_A", "wilcoxon_statistic",
                "p_value", "rank_biserial_effect", "p_holm"]:
        print(f"max |new - frozen| {col:28s}: {(new[col] - old[col]).abs().max():.2e}")
    print("Same Holm decisions:", bool((new['significant_holm'] == old['significant_holm']).all()))
    print("(bootstrap CIs may differ slightly - the original seed is unknown)")
